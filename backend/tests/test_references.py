"""Regression tests for references.py — circular-number and citation
detection. Pure functions, no database needed.

Run with:  pytest tests/test_references.py -v
Or standalone:  python tests/test_references.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import references as r


# ---------------------------------------------------------------- Marathi detection

def test_marathi_kra_variants():
    """क्र./क्रं/क्रमांक/जा.क्र. with any punctuation flavour."""
    cases = {
        'क्र.पोमसं/१४/६६/क्रि.रि.पि.क्र. अन्वये': 'पोमसं/14/66/क्रि.रि.पि.क्र',
        'क्र.पोमसं/ २३ /२४ नुसार': 'पोमसं/23/24',
        'क्रमांक:- पोमसं/२१/७२७२ प्रमाणे': 'पोमसं/21/7272',
        'जा.क्र. पोमसं/२२/११ अन्वये': 'पोमसं/22/11',
        'क्र.  पोमसं/२५/९९ पहा': 'पोमसं/25/99',
        'क्रं. 77/2022 पहा': '77/2022',
    }
    for text, expect in cases.items():
        got = [n for _, n in r.extract_references(text)]
        assert expect in got, f"{text!r} -> {got}"


def test_marathi_spaced_and_bare_forms():
    cases = {
        'पोमसं १४/६६ अन्वये कार्यवाही': 'पोमसं14/66',
        'पो.म.सं. १४/६६/२०२३ नुसार': 'पो.म.सं.14/66/2023',
        '५२५/२०२३ अन्वये कार्यवाही': '525/2023',
        'सदर 14/66/2023 नुसार': '14/66/2023',
        '१२/२०२३ नुसार अंमलबजावणी': '12/2023',
    }
    for text, expect in cases.items():
        got = [n for _, n in r.extract_references(text)]
        assert expect in got, f"{text!r} -> {got}"


def test_multiword_devanagari_phrase_in_identifier():
    """A genuine two-word Marathi phrase (e.g. 'साखळी चोरी' = 'chain theft')
    inside a circular number must be captured whole, not truncated at the
    space."""
    text = "क्र पोमसं/२३/५४/साखळी चोरी/ ३८० /२०१६"
    own = r.detect_own_number(text, "14_28.pdf")
    assert own == "पोमसं/23/54/साखळीचोरी/380/2016", own


def test_multiword_phrase_does_not_swallow_trailing_prose():
    """Multi-word tolerance must not consume ordinary connector words that
    happen to follow a number (e.g. '...2016 अन्वये कारवाई करावी')."""
    cases = [
        "क्र. पोमसं/22/2016 अन्वये कारवाई करावी.",
        "क्र. पोमसं/45/2020 नुसार अंमलबजावणी करावी.",
        "जा.क्र. 123/2019 बाबत सविस्तर अहवाल पाठवावा.",
    ]
    for text in cases:
        refs = [n for _, n in r.extract_references(text)]
        assert not any(w in x for x in refs for w in ("अन्वये", "नुसार", "बाबत")), (text, refs)


def test_own_number_and_citations_full_document():
    """Full real-document check: own number plus two genuine citations,
    with no fragment leakage from the multi-word segment."""
    text = """क्र पोमसं/२३/५४/साखळी चोरी/ ३८० /२०१६
संदर्भ: - या कार्यालयाचे क्र. पोमसं/२३/५४/साखळी चोरी/२६८/२०११
       दि. ०१/३/२०११ व पोमसं/२३/५४/साखळी चोरी/४५०/२०११,"""
    own = r.detect_own_number(text, "14_28.pdf")
    refs = [n for _, n in r.split_body_references(text, own)]
    assert own == "पोमसं/23/54/साखळीचोरी/380/2016"
    assert set(refs) == {"पोमसं/23/54/साखळीचोरी/268/2011",
                         "पोमसं/23/54/साखळीचोरी/450/2011"}


def test_gazette_false_positive_fixed():
    """'पत्र' (letter) must not match as a substring inside 'राजपत्र'
    (Gazette) — this previously caused 'राजपत्र भाग-2' to be wrongly
    captured as if 'पत्र' were a label and 'भाग-2' its identifier."""
    text = "भारत सरकारचा सुधारणा कायदा राजपत्र भाग-२ खंड-१, दिनांक ०१.०९.२००० व महाराष्ट्र शासन"
    refs = [n for _, n in r.extract_references(text)]
    assert "भाग-2" not in refs

    # genuine "पत्र" (letter) as its own word must still work; "सं" here is
    # correctly stripped as a redundant label prefix (same as क्र), leaving
    # the clean identifier
    assert '45/2023' in [n for _, n in
                         r.extract_references('शासन पत्र क्र. सं/45/2023 नुसार')]


# ---------------------------------------------------------------- English detection

def test_english_structured_citations():
    cases = {
        'refer Circular No. 12/2024 for details': '12/2024',
        'Circular Number 45-B/2020 applies': '45-B/2020',
        'F.No. 12/34/2023-Admn': '12/34/2023-ADMN',
        'Notification No. S.O.1234(E) issued': 'S.O.1234(E)',
    }
    for text, expect in cases.items():
        got = [n for _, n in r.extract_references(text)]
        assert expect in got, (text, got)


def test_own_number_label_stripped_cleanly():
    """A document's own number ('Circular No.42') must be stored WITHOUT
    the redundant 'NO.' prefix leaking in."""
    text = ("INSPECTOR GENERAL'S STANDING ORDER CIRCULAR NO.42\n"
            "Circular No.42 (5132), Poona, dated 25th June, 1952.")
    own = r.detect_own_number(text, "17_1.pdf")
    assert own == "42", own


def test_overly_generic_english_citation_excluded():
    """A bare, unstructured citation ('Circular No.1', no slash/year/dept
    code) is too generic to be an actionable missing-reference flag and
    must be excluded — but only as a CITATION, not as an own-number."""
    text = ("INSPECTOR GENERAL'S STANDING ORDER CIRCULAR NO.42\n"
            "Circular No.42 (5132), Poona, dated 25th June, 1952.\n"
            "In supersession of Inspector-General's Circular No.1, dated "
            "17th January 1941, published in Part (b) of the Bombay Police "
            "Guzette, dated 23rd January 1941.")
    own = r.detect_own_number(text, "17_1.pdf")
    refs = r.split_body_references(text, own)
    assert own == "42"
    assert refs == [], refs


def test_rbi_style_regression():
    text = ('RESERVE BANK OF INDIA\nRBI/2024-25/102\n'
           'refer Circular No. 12/2023 and RBI/2022-23/45')
    own = r.detect_own_number(text)
    assert own == 'RBI/2024-25/102'
    refs = [n for _, n in r.split_body_references(text, own)]
    assert set(refs) == {'12/2023', 'RBI/2022-23/45'}


# ---------------------------------------------------------------- noise rejection

def test_dates_and_noise_rejected():
    noise = [
        'दिनांक १५/०३/२०२३ रोजी', 'दि. १२/२०२३ रोजीच्या पत्रान्वये',
        'दि.१२/२०२३ रोजी', 'dated 12/2023 regarding leave',
        'बैठक 31/12/2025 रोजी', 'बैठक ३१/१२/२०२५ रोजी',
        'गुण 14/66 मिळाले', 'पृष्ठ १/२ पहा', 'कलम ३/४ नुसार',
    ]
    for text in noise:
        assert [n for _, n in r.extract_references(text)] == [], text


# ---------------------------------------------------------------- fuzzy matching

def test_fuzzy_reference_matching():
    assert r.refs_match('12/2023', '12/2023')          # exact
    assert r.refs_match('12/2023', 'GAD/12/2023')       # suffix
    assert not r.refs_match('12/2023', '12/2024')       # different year, no match
    assert r.refs_match('पोमसं14/66', 'पोमसं/14/66')     # skeleton (separator-free)


def test_bare_no_prefix_with_no_label_word():
    """British-era standing orders sometimes state their own number as
    just 'No. G/3239' with no preceding label word (Circular/F/
    Notification/etc.) -- confirmed via a real failing document. Must be
    detected, while a plain series number like 'Standing Order No. 145'
    must NOT be mistaken for it (line-start anchor + _LAT's structural
    minimum + distinctiveness check all guard this together)."""
    text = """DGP STANDING ORDERS/CIRCULARS/CRIME
STANDING ORDER No. 145.
No. G/3239, Bombay 13th July 1965
Subject: Conference of the Police Officers"""
    own = r.detect_own_number(text, "18_19.pdf")
    assert own == "G/3239", own


def test_bare_no_prefix_false_positive_guards():
    """'No.'/'no.' appearing mid-sentence in ordinary prose must never be
    mistaken for a circular number -- this pattern is inherently risky
    (an extremely common word) so needs real scrutiny."""
    negative_cases = [
        "There are no. of applicants waiting for the exam.",
        "See page no. 12 for details.",
        "Room no. 5 is reserved for the meeting.",
        "Serial no. 3 in the table below.",
        "Rule no. 6(2) applies here.",
    ]
    for text in negative_cases:
        got = [n for _, n in r.extract_references(text)]
        assert got == [], (text, got)


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
