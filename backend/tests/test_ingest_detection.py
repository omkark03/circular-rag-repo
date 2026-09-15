"""Regression tests for ingest.py — OCR quality detection and metadata
guessing. Pure functions, no database or real Tesseract/model needed.

Run with:  pytest tests/test_ingest_detection.py -v
Or standalone:  python tests/test_ingest_detection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import ingest


# ---------------------------------------------------------------- OCR language mismatch

def test_garbled_ocr_flagged_as_suspect():
    """English-forced OCR on a Devanagari document produces plausible-
    looking Latin gibberish — this must be caught, not silently indexed."""
    config.OCR_LANG = "mar+eng"
    garbled = ("ARTS / Herat WH. WA/2HAT-BS-AATH/ 200, Tag, few - %v.20,2000 "
              "fasa- ight uedien ater arevtaad aaa dat Sistah oreaifae " * 3)
    assert ingest.ocr_language_mismatch(garbled) is True


def test_genuine_marathi_not_flagged():
    config.OCR_LANG = "mar+eng"
    real = ("महाराष्ट्र शासन सामान्य प्रशासन विभाग परिपत्रक क्रमांक संकीर्ण-2023 "
           "दिनांक पंधरा मार्च दोन हजार तेवीस रोजी निर्गमित करण्यात येत आहे")
    assert ingest.ocr_language_mismatch(real) is False


def test_english_only_deployment_never_flags():
    """If OCR_LANG doesn't declare Devanagari expected, the check must not
    fire regardless of content — it can't second-guess a deployment
    genuinely configured for English-only OCR."""
    config.OCR_LANG = "eng"
    garbled = "ARTS / Herat WH. WA/2HAT-BS-AATH/ 200 " * 5
    assert ingest.ocr_language_mismatch(garbled) is False
    config.OCR_LANG = "mar+eng"  # restore for subsequent tests


def test_short_text_not_judged():
    config.OCR_LANG = "mar+eng"
    assert ingest.ocr_language_mismatch("short bit") is False


def test_legitimate_english_attachment_not_flagged():
    """A document that legitimately contains a large genuinely-English
    attachment (e.g. a court judgment enclosed with a Marathi circular)
    must NOT be confused with actual OCR gibberish — even though its
    overall Devanagari ratio is naturally low."""
    config.OCR_LANG = "mar+eng"
    judgment = """IN THE HIGH COURT OF JUDICATURE AT BOMBAY
NAGPUR BENCH AT NAGPUR
CRIMINAL APPLICATION (APL) No. 91 OF 2015
CORAM: A.B.CHAUDHARI AND P.N. DESHMUKH, JJ.
This is an application under Section 482 of Code of Criminal Procedure. The
application is liable to be thrown out at the threshold. However, looking to
the fact that the applicant is 82 years old person and by the preliminary
order made against him in Case No.115/2014 by the Executive Magistrate, we
have considered the matter. The contents of the said order obviously show
exercise of power under Section 116(3) of the Code of Criminal Procedure."""
    assert ingest.ocr_language_mismatch(judgment) is False


def test_gibberish_still_caught_even_though_english_check_exists():
    """The English-prose exception must not accidentally let real gibberish
    through — gibberish doesn't read as genuine English prose either."""
    config.OCR_LANG = "mar+eng"
    garbled = ("WH. WA/2HAT-BS-AATH fasa ight uedien ater arevtaad aaa dat "
              "Sistah oreaifae RAS BOLT uftaae Farge afafray 2s42 " * 3)
    assert ingest.ocr_language_mismatch(garbled) is True


# ---------------------------------------------------------------- metadata guessing

def test_running_header_skipped_for_issuer():
    """A line repeating across every page (e.g. a running page header) must
    not be picked as the issuer — the next real content line should be
    used instead."""
    header = "DGP STANDING ORDERS/CIRCULARS/INVESTIGATION"
    page_body = ("INSPECTOR GENERAL'S STANDING ORDER CIRCULAR NO.42\n"
                "Circular No.42 (5132), Poona, dated 25th June, 1952.\n")
    text = ("\n" + header + "\n" + page_body) + (header + "\npage content\n") * 18
    meta = ingest._guess_metadata(text, "17_1.pdf")
    assert meta["issuer"] != header
    assert "CIRCULAR" in meta["issuer"].upper() or "INSPECTOR" in meta["issuer"].upper()


def test_normal_first_line_issuer_unaffected():
    """A document without any repeated header should use the first line as
    issuer, unaffected by the running-header check."""
    text = "महाराष्ट्र शासन\nशासन परिपत्रक क्रमांक: संकीर्ण-2023/45\nविषय :- चाचणी परिपत्रक"
    meta = ingest._guess_metadata(text, "test.pdf")
    assert meta["issuer"] == "महाराष्ट्र शासन"


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
