"""Regression tests for OCR pre-processing (deskew + contrast enhancement)
in ingest.py. Uses real OpenCV against synthetic images with known
rotation angles — no Tesseract or real scanned documents needed.

Run with:  pytest tests/test_ocr_preprocessing.py -v
Or standalone:  python tests/test_ocr_preprocessing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from PIL import Image

import config
import ingest


def _make_skewed_page(angle_deg, size=400, bg=255, fg=0):
    img = np.full((size, size), bg, dtype=np.uint8)
    for y in range(50, 350, 30):
        cv2.rectangle(img, (40, y), (360, y + 12), fg, -1)
    center = (size // 2, size // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, m, (size, size), borderValue=bg)


def test_skew_detection_recovers_known_rotation_angle():
    for true_angle in [-8, -3, 3, 8]:
        page = _make_skewed_page(true_angle)
        detected = ingest._detect_skew_angle(page)
        # detection recovers the CORRECTION angle -- the negative of how
        # far the content was rotated away from upright
        assert abs(detected - (-true_angle)) < 2.0, \
            f"true={true_angle}, detected={detected}"


def test_skew_detection_near_zero_for_unrotated_page():
    page = _make_skewed_page(0)
    detected = ingest._detect_skew_angle(page)
    assert abs(detected) < 2.0


def test_skew_detection_handles_near_empty_image():
    """A blank/near-empty page has too little content to estimate skew
    reliably -- must return 0.0, not crash or produce a wild guess."""
    blank = np.full((400, 400), 255, dtype=np.uint8)
    assert ingest._detect_skew_angle(blank) == 0.0


def test_preprocess_widens_contrast_range():
    """CLAHE contrast enhancement should measurably widen the dynamic
    range on a faded/low-contrast synthetic page."""
    page = _make_skewed_page(6, bg=200, fg=100)  # faded: 100-200, not 0-255
    original = Image.fromarray(page).convert("RGB")
    processed = ingest._preprocess_image(original)

    assert processed.size == original.size
    orig_range = np.ptp(np.array(original.convert("L")))
    proc_range = np.ptp(np.array(processed))
    assert proc_range > orig_range, \
        f"contrast should widen: {orig_range} -> {proc_range}"


def test_preprocess_gracefully_degrades_without_opencv():
    """If OpenCV isn't installed, pre-processing must be skipped cleanly
    (return the original image unchanged), never crash the OCR pipeline
    over an optional enhancement."""
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated: opencv not installed")
        return real_import(name, *args, **kwargs)

    original = Image.new("RGB", (100, 100), "white")
    builtins.__import__ = blocking_import
    try:
        result = ingest._preprocess_image(original)
        assert result is original
    finally:
        builtins.__import__ = real_import


def test_psm_oem_passed_through_to_tesseract_call():
    """Confirms the configured PSM/OEM values actually reach the real
    pytesseract call, not just exist in config unused."""
    import sys as _sys
    from unittest.mock import MagicMock

    fake_pytesseract = MagicMock()
    fake_pytesseract.image_to_string.return_value = "extracted text"
    _sys.modules["pytesseract"] = fake_pytesseract
    try:
        old_preprocess, old_psm, old_oem, old_lang = (
            config.OCR_PREPROCESS, config.OCR_PSM, config.OCR_OEM, config.OCR_LANG)
        config.OCR_PREPROCESS = False  # isolate to just the tesseract-call args
        config.OCR_PSM = 6
        config.OCR_OEM = 1
        config.OCR_LANG = "mar+eng"

        img = Image.new("RGB", (100, 100), "white")
        ingest._ocr_image(img)

        call_kwargs = fake_pytesseract.image_to_string.call_args.kwargs
        assert call_kwargs["lang"] == "mar+eng"
        assert call_kwargs["config"] == "--psm 6 --oem 1"
    finally:
        config.OCR_PREPROCESS, config.OCR_PSM, config.OCR_OEM, config.OCR_LANG = (
            old_preprocess, old_psm, old_oem, old_lang)
        del _sys.modules["pytesseract"]


def test_preprocess_falls_back_on_processing_error_not_just_missing_import():
    """Regression: a real image tripping up cv2 mid-processing (not just
    cv2 being uninstalled) must still fall back gracefully to the
    unprocessed image, not crash the whole upload."""
    import cv2
    real_cvtColor = cv2.cvtColor
    cv2.cvtColor = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("simulated: unexpected image format"))
    try:
        img = Image.new("RGB", (100, 100), "white")
        result = ingest._preprocess_image(img)
        assert result is img
    finally:
        cv2.cvtColor = real_cvtColor


def test_ocr_retries_with_defaults_when_tuned_psm_oem_fails():
    """Regression: OCR_OEM=1 (LSTM-only) can fail entirely if the
    installed traineddata doesn't include LSTM data -- a real Tesseract
    failure mode, not hypothetical. Must retry with plain defaults
    instead of failing the whole upload."""
    import sys as _sys
    from unittest.mock import MagicMock

    fake_pytesseract = MagicMock()
    call_log = []

    def fake_image_to_string(img, lang=None, config=None):
        call_log.append({"lang": lang, "config": config})
        if config is not None:
            raise Exception("TesseractError: LSTM requested, but not present")
        return "fallback OCR text"

    fake_pytesseract.image_to_string.side_effect = fake_image_to_string
    _sys.modules["pytesseract"] = fake_pytesseract
    try:
        old_preprocess, old_psm, old_oem, old_lang = (
            config.OCR_PREPROCESS, config.OCR_PSM, config.OCR_OEM, config.OCR_LANG)
        config.OCR_PREPROCESS = False
        config.OCR_PSM = 3
        config.OCR_OEM = 1
        config.OCR_LANG = "mar+eng"

        result = ingest._ocr_image(Image.new("RGB", (100, 100), "white"))
        assert result == "fallback OCR text"
        assert len(call_log) == 2
        assert call_log[0]["config"] == "--psm 3 --oem 1"
        assert call_log[1]["config"] is None
    finally:
        config.OCR_PREPROCESS, config.OCR_PSM, config.OCR_OEM, config.OCR_LANG = (
            old_preprocess, old_psm, old_oem, old_lang)
        del _sys.modules["pytesseract"]


def test_ocr_error_propagates_when_even_fallback_fails():
    """When the retry ALSO fails (e.g. Tesseract genuinely isn't
    installed), the real error must still surface -- not be silently
    swallowed just because we added a fallback attempt."""
    import sys as _sys
    from unittest.mock import MagicMock

    fake_pytesseract = MagicMock()
    fake_pytesseract.image_to_string.side_effect = Exception("Tesseract not found entirely")
    _sys.modules["pytesseract"] = fake_pytesseract
    try:
        old_preprocess = config.OCR_PREPROCESS
        config.OCR_PREPROCESS = False
        raised = False
        try:
            ingest._ocr_image(Image.new("RGB", (100, 100), "white"))
        except Exception as e:
            raised = True
            assert "Tesseract not found entirely" in str(e)
        assert raised, "the real error must propagate, not be silently lost"
    finally:
        config.OCR_PREPROCESS = old_preprocess
        del _sys.modules["pytesseract"]
def test_rename_stored_file_db_first_prevents_orphaned_rename():
    """Regression: found via real testing — the original ordering renamed
    the file on disk BEFORE updating the database. If the DB update then
    failed for any reason, the file was already moved to a path the
    database didn't know about, silently breaking every future lookup.
    DB must be updated first; the file must be left completely untouched
    if that fails."""
    import store as _store

    real_update = _store.update_document
    _store.update_document = lambda *a, **kw: (_ for _ in ()).throw(
        Exception("simulated DB failure"))
    try:
        config.FILES_DIR.mkdir(parents=True, exist_ok=True)
        old_name = "test999_orig.pdf"
        (config.FILES_DIR / old_name).write_bytes(b"content")

        result = ingest._rename_stored_file(999, old_name, "Some Title", "orig.pdf")

        assert result == old_name, "must return the original name on DB failure"
        assert (config.FILES_DIR / old_name).exists(), \
            "file must be left untouched when the DB update fails"
        assert not (config.FILES_DIR / "test999_Some_Title.pdf").exists(), \
            "file must NOT be renamed when the DB update fails"
    finally:
        _store.update_document = real_update
        (config.FILES_DIR / old_name).unlink(missing_ok=True)
def test_translation_venv_default_path_is_platform_aware():
    """Regression: a real Windows deployment reported 'venv not found' at
    a bin/python3 path that never existed on that OS -- the venv had
    actually been created fine, just at Scripts/python.exe (the Windows
    convention). The default path detection must check both conventions
    and use whichever actually exists, not hardcode one."""
    import shutil
    venv_dir = config.BASE_DIR / "translation_venv"
    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        # No venv yet -- must not crash, must give an OS-appropriate guess
        result = config._default_translation_venv_python()
        assert result  # just needs to not raise and return something

        # Windows-style venv present
        (venv_dir / "Scripts").mkdir(parents=True)
        (venv_dir / "Scripts" / "python.exe").write_text("fake")
        result2 = config._default_translation_venv_python()
        assert "Scripts" in result2 and "python.exe" in result2
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)
if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)






