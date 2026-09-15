"""Detect circular/GR numbers in text and cross-check against the repository.

Robust to real-world OCR output: zero-width joiners inside conjuncts (क्‍र),
line breaks inside numbers, spaces around slashes, and space-separated
prefixes (पोमसं १४/६६). Patterns are tagged with a guard kind so ambiguous
forms (bare numerics, spaced prefixes) get extra date/keyword filtering.
"""
import re
import unicodedata

# ---------------------------------------------------------------- text cleaning

_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\ufeff]")   # ZWSP/ZWNJ/ZWJ/BOM


def clean_text(text: str) -> str:
    """NFC-normalize and strip invisible characters OCR inserts inside
    Devanagari conjuncts — 'क्‍र' (with ZWJ) becomes plain 'क्र'."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFC", text))


# ---------------------------------------------------------------- pattern pieces

_DEVL = r"\u0900-\u0965\u0970-\u097F"        # Devanagari letters (digits excluded)
_SEG = r"[A-Za-z0-9\u0900-\u097F.()\-]"      # one identifier segment char
# '/' may be padded with spaces and a single line break (OCR wraps numbers)
_SL = r"[ \t]*\n?[ \t]*/[ \t]*\n?[ \t]*"

# Common Marathi connector/verb words that follow a citation but are never
# part of the identifier itself (e.g. "...2016 अन्वये कारवाई करावी" —
# without this stoplist, a naive multi-word segment would swallow "अन्वये").
_CONNECTORS = ("अन्वये|नुसार|बाबत|करावी|करावेत|व्हावी|व्हावेत|संदर्भात|"
              "त्यानुसार|प्रमाणे|यानुसार|मुळे|साठी|तसेच|असून|असल्याने|"
              "यांना|यांचे|रोजी|दिनांक|यांच्या|याबाबत")
_NOT_CONNECTOR = r"(?!(?:" + _CONNECTORS + r")(?:[ \t.,:;)।]|$))"
_WORD = r"[A-Za-z0-9\u0900-\u097F.()\-]{1,40}"
# One path segment = 1-3 space-joined words, e.g. "साखळी चोरी". The
# continuation word must contain a Devanagari letter (real phrases do;
# stray trailing Latin/punctuation fragments don't) and must not be a
# connector word from the stoplist above.
_SEGMENT = (
    r"(?:" + _WORD + r"(?<=[" + _DEVL + r"A-Za-z])"
    r"(?:[ \t]" + _NOT_CONNECTOR + r"(?=" + _SEG + r"*[" + _DEVL + r"])" + _WORD + r"){1,2}"
    r"|" + _WORD + r")"
)
_DEV = _SEGMENT + r"(?:" + _SL + _SEGMENT + r"){0,8}"
_LAT = r"[A-Z0-9][A-Z0-9./()\-]{2,60}[A-Z0-9)]"
# क्र / क्रं / क्रमांक / सं / संख्या / जा.क्र. with any spacing/punctuation
_KRA = (r"(?:जा\s*\.?\s*)?(?:क्रमांक|क्रं|क्र|संख्या|सं)\s*\.?\s*[:\-–]*\s*")

# Each entry: (compiled pattern with one capture group, guard kind)
#   guard 'none'   -> standard filters only
#   guard 'bare'   -> also: date-context + distinctiveness checks
#   guard 'spaced' -> also: date-context + leading-word blacklist
REF_PATTERNS = [
    # RBI/2023-24/45, SEBI/HO/MIRSD/DOP/CIR/P/2023/12
    (re.compile(r"\b([A-Z]{2,6}(?:/[A-Z0-9.\-]+){1,7}/\d{2,4})\b"), "none"),
    # Circular No. ...
    (re.compile(r"\bCircular\s*(?:No\.?|Number|#)?\s*[:\-]?\s*(" + _LAT + r")",
                re.IGNORECASE), "none"),
    # F.No. / G.R. No. / Notification / Order / Letter / Memo / Ref / D.O.
    (re.compile(
        r"\b(?:F|G\.?\s?R|Government\s+Resolution|Notification|Order|Letter|"
        r"Memo(?:randum)?|Ref(?:erence)?|D\.?O)\.?\s*(?:No\.?|Number)?\s*[:\-]?\s*"
        r"(" + _LAT + r")", re.IGNORECASE), "none"),
    # शासन परिपत्रक/निर्णय/आदेश/अधिसूचना/ज्ञापन/पत्र [क्र...] <id>. The
    # lookbehind stops "पत्र" from matching inside "राजपत्र" (Gazette) —
    # without it, "...राजपत्र भाग-2..." wrongly captured "भाग-2" as if
    # "पत्र" were the label and "भाग-2" its identifier.
    (re.compile(
        r"(?<![\u0900-\u097F])(?:शासन\s*)?(?:परिपत्रक|निर्णय|आदेश|अधिसूचना|ज्ञापन|पत्र)"
        r"\s*\.?\s*(?:" + _KRA + r")?(" + _DEV + r")"), "none"),
    # standalone क्र./क्रं/क्रमांक/जा.क्र. <id> — any punctuation flavour
    (re.compile(_KRA + r"(" + _DEV + r")"), "none"),
    # bare Devanagari path, no label: पोमसं/२४/५२५२ (single-word segments
    # only — multi-word joining here is too permissive with no label to
    # anchor where the match should start, and ends up consuming preceding
    # sentence words like "क्र" or "कार्यालयाचे")
    (re.compile(
        r"((?=" + _SEG + r"*[" + _DEVL + r"])" + _SEG + r"{2,40}"
        r"(?:" + _SL + _SEG + r"{1,40}){1,8})"), "spaced"),
    # a second citation continuing a shared label via a conjunction, e.g.
    # "...क्र. X दि. ... व Y..." — Marathi commonly states one label once
    # then lists further identifiers joined by व/आणि (and); DOES use
    # multi-word segments since it's anchored by the conjunction itself.
    (re.compile(r"(?:\bव\b|आणि)\s+(" + _DEV + r")"), "none"),
    # spaced Devanagari prefix + numeric path, no slash after the word:
    #   पोमसं १४/६६  ·  पो.म.सं. १४/६६/२०२३
    (re.compile(
        r"([" + _DEVL + r"][\u0900-\u097F.]{1,20}[ \t]{1,2}\d{1,5}"
        r"(?:" + _SL + _SEG + r"{1,40}){1,8})"), "spaced"),
    # bare numeric path, no label at all: ५२५/२०२३, 14/66/2023
    (re.compile(r"(\d{1,5}(?:" + _SL + _SEG + r"{1,40}){1,8})"), "bare"),
    # Master Circular / Office Memorandum
    (re.compile(
        r"\b(?:Master\s+Circular|Office\s+Memorandum|O\.?M\.?)\s*"
        r"(?:No\.?)?\s*[:\-]?\s*(" + _LAT + r")", re.IGNORECASE), "none"),
    # bare "No. <identifier>" with NO preceding label word (Circular/F/
    # Notification/etc.) — some older/British-era standing orders state
    # their own number this way, e.g. "No. G/3239, Bombay 13th July 1965".
    # Guarded as "bare": _distinct_enough requires a letter present (or
    # 3+ segments, or a year segment) so a short plain number doesn't
    # qualify even before _LAT's own 4-character minimum already blocks
    # it — e.g. "Standing Order No. 145" does NOT match this (a series
    # number, not a citable file number), while "No. G/3239" does.
    (re.compile(r"(?:^|\n)\s*No\.?\s*[:\-]?\s*(" + _LAT + r")",
                re.IGNORECASE | re.MULTILINE), "bare"),
]

STOPWORDS = {"DATED", "ISSUED", "THE", "AND"}

# Words that start date/measure phrases, not circular numbers
# NB: re's \\b is unreliable after Devanagari combining vowels, so require
# an explicit separator+digit instead of a word boundary
_BAD_LEAD = re.compile(
    r"^(?:दिनांक|दि|सन|पृष्ठ|पान|रोजी|वेळ|वर्ष|कलम|नियम|मुद्दा|परिच्छेद|"
    r"अनुच्छेद|भाग|प्रकरण|बैठक)[\s.:\-–]*\d")

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# ---------------------------------------------------------------- normalization

def normalize(ref: str) -> str:
    """Canonical form: ASCII digits, uppercase, no whitespace, trimmed
    punctuation, glued क्र/NO.-type labels stripped, unmatched brackets
    removed. Stripping is unconditional — even a short-but-genuine own
    number like 'NO.42' -> '42' should read cleanly; short BARE numbers
    being too generic to use as a *citation* is handled separately by
    _too_generic_for_citation(), not by keeping the label glued on here."""
    ref = clean_text(ref).translate(DEVANAGARI_DIGITS).strip().upper()
    ref = re.sub(r"\s+", "", ref)
    ref = re.sub(r"^(?:जा[.:\-–]*)?(?:क्रमांक|क्रं|क्र|संख्या|सं)[.:\-–/]+", "", ref)
    ref = re.sub(r"^(?:NO\.?|NUMBER|#)[.:\-–]*", "", ref)
    ref = ref.strip(".,;:/")
    if ref.endswith(")") and "(" not in ref[:-1]:
        ref = ref[:-1]
    if ref.startswith("(") and ")" not in ref[1:]:
        ref = ref[1:]
    if ref.endswith("]") and "[" not in ref[:-1]:
        ref = ref[:-1]
    if ref.startswith("[") and "]" not in ref[1:]:
        ref = ref[1:]
    return ref


def _skeleton(norm: str) -> str:
    """Separator-free form for lenient comparison: पोमसं14/66 vs पोमसं/14/66."""
    return re.sub(r"[^A-Z0-9\u0900-\u097F]", "", norm)


# ---------------------------------------------------------------- guards

def _is_datelike(norm: str) -> bool:
    """Plausible dates only: 15/03/2023 yes; 14/66/2023 no (66 isn't a month)."""
    m = re.fullmatch(r"(\d{1,4})/(\d{1,2})/(\d{1,4})", norm)
    if not m:
        return False
    a, b, c = (int(x) for x in m.groups())
    if 1 <= a <= 31 and 1 <= b <= 12 and (c >= 1900 or c <= 99):
        return True
    if a >= 1900 and 1 <= b <= 12 and 1 <= c <= 31:
        return True
    return False


_DATE_CONTEXT = re.compile(
    r"(?:दिनांक|दि\s*\.|dated|dt\s*\.?|date\s*[:\-]?)\s*$", re.IGNORECASE)


def _looks_like_date_context(text: str, start: int) -> bool:
    return bool(_DATE_CONTEXT.search(text[max(0, start - 14):start]))


def _distinct_enough(norm: str) -> bool:
    """Bare numerics must be distinctive: a year segment, 3+ segments,
    or letters. Plain '14/66' is too ambiguous."""
    if re.search(r"[A-Za-z\u0900-\u097F]", norm):
        return True
    parts = norm.split("/")
    if len(parts) >= 3:
        return True
    return any(re.fullmatch(r"(19|20)\d{2}", p) for p in parts)


# ---------------------------------------------------------------- extraction

def extract_references(text: str) -> list[tuple[str, str]]:
    """[(as_found, normalized)] unique by norm, ordered by first appearance.
    Fragments contained in a longer identifier are suppressed."""
    text = clean_text(text)
    found: dict[str, tuple[str, int]] = {}
    for pat, guard in REF_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1)
            norm = normalize(raw)
            if len(norm) < 2 or norm in STOPWORDS:
                continue
            if not re.search(r"\d", norm):
                continue
            if _is_datelike(norm):
                continue
            if guard in ("bare", "spaced") and _looks_like_date_context(text, m.start()):
                continue
            if guard == "bare" and not _distinct_enough(norm):
                continue
            if guard == "spaced":
                if _BAD_LEAD.match(raw.strip()):
                    continue
                tail_m = re.search(r"(\d.*)$", norm)
                tail = tail_m.group(1) if tail_m else ""
                # word glued to a date (बैठक 31/12/2025)
                if _is_datelike(tail):
                    continue
                # only a SPACE joins prefix and number: demand structure —
                # prefix >=4 chars (पोमसं yes, गुण no) and a >=2-digit segment
                # in the tail (14/66 yes, 3/4 no)
                if " " in raw:
                    prefix = re.match(r"^[\u0900-\u097F.]+", norm)
                    if not prefix or len(prefix.group(0).rstrip(".")) < 4:
                        continue
                    if not re.search(r"\d{2,}", tail):
                        continue
            if norm not in found or m.start() < found[norm][1]:
                found[norm] = (raw.strip(), m.start())

    norms = sorted(found, key=len, reverse=True)
    keep = []
    for n in norms:
        if any(n != k and n in k for k in keep):
            continue
        keep.append(n)

    ordered = sorted(((n, found[n]) for n in keep), key=lambda kv: kv[1][1])
    return [(raw, norm) for norm, (raw, _pos) in ordered]


def detect_own_number(text: str, filename: str = "") -> str | None:
    """Earliest identifier in the letterhead area, else from the filename."""
    refs = extract_references(text[:1500])
    if refs:
        return refs[0][1]
    refs = extract_references(filename.replace("_", "/"))
    if refs:
        return refs[0][1]
    return None


_GENERIC_BARE_NUMBER = re.compile(r"^(?:NO\.?|NUMBER|#)?\d{1,3}$")


def _too_generic_for_citation(norm: str) -> bool:
    """True for bare short numbers with no other structure — e.g. a citation
    reading just 'Circular No. 1' with no slash, year, or department code.
    These are usually historical/superseded references with nothing else to
    anchor them, and flagging them as "missing" is low-signal noise rather
    than an actionable citation. A document's OWN number is never checked
    against this — only candidates being considered as citations TO other
    documents."""
    return bool(_GENERIC_BARE_NUMBER.match(norm))


def split_body_references(text: str, own_number: str | None) -> list[tuple[str, str]]:
    out = []
    for raw, norm in extract_references(text):
        if own_number and (norm == own_number or norm in own_number):
            continue
        if _too_generic_for_citation(norm):
            continue
        out.append((raw, norm))
    return out


# ---------------------------------------------------------------- matching

def refs_match(a: str, b: str) -> bool:
    """Conservative fuzzy match: exact, suffix, distinctive containment,
    or separator-free skeleton equality (पोमसं14/66 ≡ पोमसं/14/66)."""
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6:
        if a.endswith("/" + b) or b.endswith("/" + a):
            return True
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 8 and shorter in longer:
            return True
        sa, sb = _skeleton(a), _skeleton(b)
        if sa == sb and len(sa) >= 8:
            return True
    return False
