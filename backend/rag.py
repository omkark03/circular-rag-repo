"""Retrieval-augmented answering with Phi-3 via Ollama.

Runs CPU-only by default; GPU offload is controlled entirely from
config (OLLAMA_NUM_GPU / EMBED_DEVICE) — no code changes needed.
References are returned as structured objects so the frontend can render
APA style with a clickable link.
"""
import json
import re

import requests

import config
import ingest
import retrieval
import store
import translation

_CITATION_RULE = """Each excerpt is preceded by a line starting with ">> \
EXCERPT_METADATA" giving issuer=, year=, and file= values. That metadata \
line is NOT part of the document and must NEVER be printed, quoted, or \
paraphrased in your output — no "EXCERPT_METADATA", no "issuer=", no ">>" \
or "<<" symbols, no filename, ever appearing in your output. When you \
state a fact, look at the specific excerpt supporting it, read the actual \
text written after issuer= in its metadata line and the actual text \
written after year=, and write those two real values inside ordinary \
parentheses as your citation — nothing bracketed, nothing mentioning \
"source" or "excerpt" or a filename. NEVER cite an issuer or year that \
isn't literally the real value from an issuer=/year= field actually \
given to you in THIS prompt — not one recalled from training, not a \
generic-sounding name you make up. This instruction itself contains no \
example citation text anywhere, on purpose — if any citation-shaped text \
ever seems to come from these instructions rather than an actual \
issuer=/year= value you just read, that means something went wrong and \
you must not write it."""


SYSTEM_PROMPT = f"""You are an assistant that answers questions strictly from \
the excerpts of official circulars provided below. Rules:
1. Answer only from the provided excerpts. If they do not contain the answer, \
say so plainly — never invent circular numbers, dates or provisions.
2. {_CITATION_RULE}
3. Write ONE direct, unified answer to the question. Do NOT structure your \
answer as a list of what each individual source states (e.g. "Circular A \
states X. Circular B states Y. Circular C states Z.") unless the question \
specifically asks you to compare sources — this is the single most common \
way this kind of answer goes wrong, so treat it as the default failure \
mode to actively avoid. A SEPARATE process already checks for and reports \
genuine contradictions between sources, so your answer does not need to \
enumerate differences between sources itself — just give the clearest, \
most direct answer to the actual question, weaving citations into normal \
prose as you go (e.g. "Inspections must happen at least once a year \
(Maharashtra Police, 1966)."), not as a running list of per-source \
statements. If multiple excerpts genuinely support the same single fact, \
cite them together once, e.g. (X, 2020; Y, 2021) — never repeat the same \
sentence or claim once per source just because multiple excerpts mention it.
4. Quote clause and sub-clause numbers precisely where present.
5. Answer in the SAME language the question was asked in (English question \
→ English answer; Marathi question → Marathi answer). When the source \
excerpts are in a different language/script than the question, ground your \
answer directly in what the excerpt actually says — do not silently \
translate the excerpt into English in your own reasoning and then \
translate that back into the answer language, since each translation step \
introduces drift and errors. Numbers, dates, circular numbers, section/ \
clause numbers, and proper names must be copied exactly as they appear in \
the source, unchanged, regardless of which language you are answering in.
6. DO NOT artificially shorten the answer. If the excerpts contain multiple \
provisions, conditions, exceptions, steps, eligibility criteria, monetary \
limits, timelines, or sub-clauses relevant to the question, explain EACH \
ONE — do not summarize several provisions into one line. Use separate \
paragraphs or a numbered/bulleted list to cover them individually and \
completely. A question that only has a one-line answer in the source may \
get a one-line answer; a question whose source material is long and \
detailed must get an equally long and detailed answer. Only skip a detail \
if it is genuinely irrelevant to the question asked.
7. Start directly with the answer — no preamble like "Based on the provided \
excerpts" and no closing summary sentence/paragraph at the end restating \
what was already said. If you are tempted to add one, stop instead — the \
answer is complete once the last real point has been made. NEVER write a \
citation using placeholder letters like (X, 1966; Y, 1981) as a stand-in \
when you can't recall which specific source went with which fact from a \
summary you're compressing — this happens most often in exactly this kind \
of closing-summary sentence, which is one more reason not to write one.
8. If the answer involves comparing multiple items, or listing rates, \
limits, slabs, dates, or eligibility criteria across categories/years/ \
circulars, format that part as a GitHub-flavored Markdown table (header \
row, |---| separator row, data rows) instead of prose. Use a table only \
when there are genuinely multiple rows and columns to compare — a single \
fact or a short list should stay as text. EVERY table row — the header, \
the |---| separator, and each data row — MUST be on its own line with a \
real line break before and after it. Never place two rows, or a row and \
the separator, on the same line.
"""


def _short_issuer(issuer: str) -> str:
    return (issuer or "Unknown issuer").split(",")[0].strip()[:80]


def build_apa_reference(doc_row) -> dict:
    issuer = _short_issuer(doc_row["issuer"])
    year = doc_row["year"] or "n.d."
    title = doc_row["title"] or doc_row["filename"]
    url = f"{config.PUBLIC_FILES_URL}/{doc_row['stored_name']}"
    apa = f"{issuer}. ({year}). {title}"
    if doc_row["circular_no"]:
        apa += f" (Circular No. {doc_row['circular_no']})"
    return {"issuer": issuer, "year": year, "title": title, "filename": doc_row["filename"],
            "circular_no": doc_row["circular_no"], "url": url, "apa": apa}


def retrieve(question: str, k: int = config.TOP_K):
    """Hybrid (vector + BM25) retrieval with cross-encoder reranking."""
    return retrieval.retrieve(question, top_k=k)


# Text that should NEVER legitimately appear in a real answer — if it does,
# the model is parroting prompt/instruction text instead of real citation
# values (this exact failure pattern recurred three times with three
# different literal strings we tried using as examples, so prompt wording
# alone isn't fully reliable against it — this is a visibility net, not a
# fix, since correcting the text automatically would be its own can of worms).
_LEAK_MARKERS = ("EXCERPT_METADATA", "ISSUER-PLACEHOLDER", "19XX",
                 "ISSUER-X", "Mumbai Police", "Reserve Bank of India")


def _check_for_leaked_markers(answer_text: str):
    found = [m for m in _LEAK_MARKERS if m in answer_text]
    if found:
        print(f"[rag] WARNING: answer contains leaked prompt/placeholder "
             f"text: {found}. The model is echoing instruction text instead "
             f"of real citation values — treat any citations in this answer "
             f"as unverified.", flush=True)
    return found


def _ollama_options() -> dict:
    options = {"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED,
               "num_gpu": config.OLLAMA_NUM_GPU,
               "num_predict": config.NUM_PREDICT, "num_ctx": config.NUM_CTX}
    if config.LLM_THREADS:
        options["num_thread"] = config.LLM_THREADS
    return options


CONTRADICTION_PROMPT = """You are a careful auditor comparing excerpts from \
different official circulars. Below are excerpts, each preceded by a \
metadata line giving its issuer=, year=, and file=.

Your ONLY task: find places where two or more excerpts state DIFFERENT or \
CONFLICTING requirements, durations, amounts, deadlines, or procedures for \
what appears to be the SAME specific matter — for example, one excerpt \
requiring a minimum of two days and another requiring a minimum of four \
days for what both describe as the same type of inspection. Do not flag \
differences that are naturally about DIFFERENT matters — only flag genuine \
conflicts about the SAME matter. A later circular updating/superseding an \
earlier one on the same point still counts as worth flagging, since the \
reader should see both and judge which currently applies.

Respond with ONLY a JSON array, nothing else — no explanation, no markdown \
code fences, no text before or after it. Each element must have exactly \
this shape:
{"topic": "short description of what they disagree about", "positions": \
[{"issuer": "...", "year": "...", "claim": "what this source states, in \
your own words"}, {"issuer": "...", "year": "...", "claim": "what this \
other source states"}]}

For issuer and year in each position, use the REAL values from the \
issuer=/year= fields in the metadata line above the excerpt supporting \
that position — never invent or guess an issuer/year that isn't literally \
given to you below.

If you find no genuine contradictions, respond with exactly: []
"""


SCHEDULE_RULES_PROMPT = """You are analyzing circular excerpts to extract \
SPECIFIC numeric inspection-scheduling rules. Below are excerpts, each \
preceded by a metadata line giving issuer=, year=, and file=.

Find and extract:
- How many days (and/or nights) an inspection should last for a REGULAR \
(not specially marked "important") police station.
- How many days (and/or nights) an inspection should last for an \
IMPORTANT police station (if the excerpts distinguish this at all).
- How many times per year each station should be inspected (default 1 \
if not stated).

Respond with ONLY a JSON object, nothing else — no explanation, no \
markdown code fences, no text before or after it. Exact shape:
{"regular_duration_days": <integer, or null if not found in the \
excerpts>, "important_duration_days": <integer, or null if not found>, \
"times_per_year": <integer, default 1 if not stated>, "source_note": \
"brief note on which circular(s) this came from, using the real \
issuer/year values from the metadata lines above"}

If you cannot find a SPECIFIC numeric duration anywhere in the excerpts, \
use null for that field — do NOT guess a plausible-sounding number.
"""


def _safe_positive_int(v, fallback=None):
    if isinstance(v, bool):
        return fallback
    if isinstance(v, int):
        return v if v > 0 else fallback
    if isinstance(v, str):
        try:
            iv = int(v.strip())
            return iv if iv > 0 else fallback
        except Exception:
            return fallback
    return fallback


def _parse_schedule_rules_json(raw: str) -> dict:
    """Robustly parse the rule-extraction JSON — same defensive handling
    as _parse_contradictions_json (code fences, stray prose, malformed
    output). Degrades to all-None/default rather than crashing OR
    inventing a plausible-sounding number; the caller must handle a
    missing duration explicitly, never silently substitute a guess."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    default = {"regular_duration_days": None, "important_duration_days": None,
              "times_per_year": 1, "source_note": ""}

    data = None
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                pass
    if not isinstance(data, dict):
        return default

    return {
        "regular_duration_days": _safe_positive_int(data.get("regular_duration_days")),
        "important_duration_days": _safe_positive_int(data.get("important_duration_days")),
        "times_per_year": _safe_positive_int(data.get("times_per_year"), fallback=1) or 1,
        "source_note": data.get("source_note") if isinstance(data.get("source_note"), str) else "",
    }


_SCHEDULE_INTENT_WORDS = ("schedule", "inspection plan", "inspection programme",
                         "inspection program")
_STATION_HINT_WORDS = ("station", "thane", "\u0920\u093e\u0923\u0947")  # ठाणे


def looks_like_schedule_request(message: str) -> bool:
    """Lightweight keyword heuristic deciding whether it's worth spending
    an LLM call on schedule extraction at all -- keeps ordinary chat
    messages exactly as fast as before. A false negative just means the
    message is answered normally (a safe failure mode); a false positive
    costs one extra fast JSON-extraction call before falling through to
    normal chat when it finds no stations."""
    text = message.lower()
    return (any(w in text for w in _SCHEDULE_INTENT_WORDS)
           and any(w in text for w in _STATION_HINT_WORDS))


SCHEDULE_REQUEST_PROMPT = """Extract inspection-scheduling information from \
the user's message below. Respond with ONLY a JSON object, nothing else — \
no explanation, no markdown code fences, no text before or after it. Use \
this exact shape:

{"stations": [{"name": "...", "important": true or false}, ...], \
"start_date": "YYYY-MM-DD" or null, "details": "..."}

Rules:
- stations: list every specific police station NAME actually mentioned in \
the message, in the order mentioned. NEVER invent a station that was not \
named. If no specific station names are mentioned, use an empty list.
- For each station, set "important" to true ONLY if the message explicitly \
describes that particular station as important/priority/major/senior — \
never guess this.
- start_date: if the message states or clearly implies a specific start \
date, convert it to YYYY-MM-DD using {today} as "today" for any relative \
reference (e.g. "next month", "starting Monday"). If no date is given, or \
it is too vague to convert with real confidence, use null — never guess a \
date.
- details: any other real scheduling context mentioned (district, special \
instructions), or an empty string if none.

If the message is not actually requesting a schedule with specific named \
stations, respond with exactly: {"stations": [], "start_date": null, "details": ""}
"""


def _parse_schedule_request_json(raw: str) -> dict:
    """Same defensive parsing pattern as _parse_schedule_rules_json —
    code fences, stray prose, malformed output all handled. Degrades to
    an empty result (no stations, no date) rather than crashing or
    inventing plausible-looking data."""
    empty = {"stations": [], "start_date": None, "details": ""}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    data = None
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                pass
    if not isinstance(data, dict):
        return empty

    stations = []
    raw_stations = data.get("stations")
    if isinstance(raw_stations, list):
        for s in raw_stations:
            if isinstance(s, dict):
                name = s.get("name")
                if isinstance(name, str) and name.strip():
                    stations.append({"name": name.strip(),
                                    "important": bool(s.get("important"))})

    start_date = data.get("start_date")
    if not (isinstance(start_date, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date)):
        start_date = None

    details = data.get("details") if isinstance(data.get("details"), str) else ""

    return {"stations": stations, "start_date": start_date, "details": details}


def extract_schedule_request(message: str) -> dict:
    """Narrow JSON-only extraction of stations/date/details from the
    USER'S OWN chat message — distinct from extract_schedule_rules, which
    extracts DURATION rules from retrieved circular excerpts. Returns an
    empty result ({"stations": [], "start_date": None, "details": ""}) on
    any failure — the caller must never proceed as if a real schedule
    request was understood when extraction actually failed."""
    from datetime import date as _date
    empty = {"stations": [], "start_date": None, "details": ""}
    prompt = (SCHEDULE_REQUEST_PROMPT.replace("{today}", _date.today().isoformat())
             + f"\n\n=== MESSAGE ===\n{message}\n\n=== JSON ===\n")
    try:
        raw = ask_phi3(prompt)
    except Exception:
        return empty
    return _parse_schedule_request_json(raw)


def extract_schedule_rules(hits: list[dict]) -> dict:
    """Narrow, structured LLM pass to extract numeric scheduling rules
    (duration per station category, inspection frequency) from retrieved
    circular excerpts. Returns a dict where regular_duration_days /
    important_duration_days may be None if the excerpts don't specify a
    number — the caller must handle that explicitly (ask the user, or
    require an override) rather than silently guessing. Deliberately
    narrow/JSON-only rather than asking the model to also compute actual
    dates, which is far less reliable — see scheduler.py."""
    if not hits:
        return {"regular_duration_days": None, "important_duration_days": None,
               "times_per_year": 1, "source_note": "No relevant circulars found."}
    context_parts = []
    for h in hits:
        issuer = _short_issuer(h["meta"].get("issuer", ""))
        year = h["meta"].get("year", "n.d.")
        excerpt = h["text"][:config.PROMPT_CHUNK_CHARS]
        context_parts.append(
            f">> EXCERPT_METADATA issuer={issuer} | year={year} | "
            f"file={h['meta']['filename']} <<\n{excerpt}"
        )
    prompt = (SCHEDULE_RULES_PROMPT
              + "\n\n=== EXCERPTS ===\n" + "\n\n---\n\n".join(context_parts)
              + "\n\n=== JSON OBJECT ===\n")
    try:
        raw = ask_phi3(prompt)
    except Exception:
        return {"regular_duration_days": None, "important_duration_days": None,
               "times_per_year": 1, "source_note": "LLM unavailable."}
    return _parse_schedule_rules_json(raw)


def _parse_contradictions_json(raw: str) -> list[dict]:
    """Robustly extract a validated contradictions list from the model's
    raw output — handles clean JSON, JSON wrapped in markdown code fences,
    JSON with stray leading/trailing prose, and malformed individual items
    (skipped rather than failing the whole response). Returns [] on any
    failure; this is a best-effort auxiliary feature that must never break
    the main answer."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    data = None
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                pass
    if not isinstance(data, list):
        return []

    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        positions = item.get("positions")
        if not isinstance(topic, str) or not topic.strip() or not isinstance(positions, list):
            continue
        clean_positions = []
        for p in positions:
            if not isinstance(p, dict):
                continue
            issuer, year, claim = p.get("issuer"), p.get("year"), p.get("claim")
            if (isinstance(issuer, str) and issuer.strip()
                    and isinstance(claim, str) and claim.strip()
                    and isinstance(year, (str, int))):
                clean_positions.append({"issuer": issuer, "year": str(year), "claim": claim})
        if len(clean_positions) >= 2:  # need at least 2 differing positions to be a "contradiction"
            valid.append({"topic": topic, "positions": clean_positions})
    return valid


def detect_contradictions(hits: list[dict]) -> list[dict]:
    """Separate, focused LLM pass over the same retrieved excerpts,
    specifically looking for conflicting requirements between different
    circulars on the same matter. Returns [] on any failure — never a
    reason to fail the main answer."""
    if not hits:
        return []
    context_parts = []
    for h in hits:
        issuer = _short_issuer(h["meta"].get("issuer", ""))
        year = h["meta"].get("year", "n.d.")
        excerpt = h["text"][:config.PROMPT_CHUNK_CHARS]
        context_parts.append(
            f">> EXCERPT_METADATA issuer={issuer} | year={year} | "
            f"file={h['meta']['filename']} <<\n{excerpt}"
        )
    prompt = (CONTRADICTION_PROMPT
              + "\n\n=== EXCERPTS ===\n" + "\n\n---\n\n".join(context_parts)
              + "\n\n=== JSON ARRAY ===\n")
    try:
        raw = ask_phi3(prompt)
    except Exception:
        return []
    return _parse_contradictions_json(raw)


def ask_phi3(prompt: str) -> str:
    options = _ollama_options()
    try:
        r = requests.post(
            config.OLLAMA_URL,
            json={"model": config.LLM_MODEL, "prompt": prompt,
                  "stream": False, "keep_alive": config.KEEP_ALIVE,
                  "options": options},
            timeout=(10, config.LLM_TIMEOUT),
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.ConnectionError:
        return (f"⚠️ Could not reach Ollama at {config.OLLAMA_URL}. "
                f"Run `ollama serve` and `ollama pull {config.LLM_MODEL}`.")
    except requests.Timeout:
        return ("⚠️ The model timed out. On CPU, Phi-3 can be slow for long "
                "contexts — try a shorter question or enable GPU offload "
                "(OLLAMA_NUM_GPU=-1).")
    except Exception as e:
        return f"⚠️ LLM error: {e}"


def build_context(question: str, history: list[dict] | None = None):
    """Returns (prompt, references, hits) or (None, [], []) if repo empty."""
    hits = retrieve(question)
    if not hits:
        return None, [], []

    doc_ids, context_parts = [], []
    for h in hits:
        did = h["meta"]["doc_id"]
        if did not in doc_ids:
            doc_ids.append(did)
        issuer = _short_issuer(h['meta'].get('issuer', ''))
        year = h['meta'].get('year', 'n.d.')
        excerpt = h["text"][:config.PROMPT_CHUNK_CHARS]
        # Deliberately NOT shaped like a citation (no parentheses, no
        # "Source" word, no em-dash) so a weak model is less likely to
        # copy this label wholesale into its answer instead of extracting
        # just the citation values from it.
        context_parts.append(
            f">> EXCERPT_METADATA issuer={issuer} | year={year} | "
            f"file={h['meta']['filename']} << (never print this metadata "
            f"line itself; only use its values to build a citation)\n{excerpt}"
        )

    convo = ""
    for m in (history or [])[-config.HISTORY_TURNS:]:
        convo += f"{m['role'].upper()}: {m['content'][:400]}\n"

    language_override = ""
    if config.TRANSLATION_ENABLED and translation.is_devanagari_question(question):
        language_override = (
            "\n\nIMPORTANT OVERRIDE: even though the question above is in "
            "Marathi/Devanagari script, write your ENTIRE answer in English "
            "only. Do not use any Devanagari script anywhere in your answer. "
            "(It will be translated to Marathi separately by a dedicated "
            "translation step after you finish — your job here is only to "
            "produce an accurate, well-grounded ENGLISH answer.)"
        )

    prompt = (SYSTEM_PROMPT
              + "\n\n=== EXCERPTS ===\n" + "\n\n---\n\n".join(context_parts)
              + (f"\n\n=== CONVERSATION SO FAR ===\n{convo}" if convo else "")
              + f"\n\n=== QUESTION ===\n{question}{language_override}"
              + "\n\n=== ANSWER ===\n")

    refs = [build_apa_reference(store.get_document(d))
            for d in doc_ids if store.get_document(d)]
    return prompt, refs, hits


EMPTY_MSG = ("The repository is empty or nothing relevant was found. "
             "Upload circulars first.")

# Presets shown in the UI; "custom" lets the user type any document type.
TEMPLATE_PRESETS = {
    "inspection_schedule": "Inspection Schedule",
    "panchnama": "Panchnama (पंचनामा)",
    "fir": "FIR — First Information Report",
}

TEMPLATE_SYSTEM_PROMPT = f"""You are a drafting assistant that produces ready-to-use, \
fillable document templates for official/administrative use, grounded in the \
circulars provided below. Rules:
1. If the excerpts specify a prescribed format, required fields, headings, or \
clauses for this document type, follow that format exactly.
2. {_CITATION_RULE}
3. If NO excerpt actually prescribes a format for this document type, say so \
in one clear line at the top: "No format-prescribing circular was found in \
the repository for this document type — the template below is a generic, \
non-mandated draft based on standard practice and MUST be verified against \
your department's actual prescribed format before use." Then still provide a \
complete, usable draft.
4. Output the template as Markdown: a title, then clearly labeled sections \
with blank fields written as underscores or bracketed placeholders, e.g. \
"Name of Officer: ____________________" or "Date: [DD/MM/YYYY]". Use a \
Markdown table for any part that is naturally tabular (e.g. a list of items \
inspected, a schedule of dates, witness details).
5. Do not invent legal section numbers, act names, or statutory citations \
that do not appear in the excerpts — leave those as a placeholder for the \
user to fill in if not found in the repository.
6. Do not add a closing summary or meta-commentary about the task; end after \
the last section of the template.
"""


SCHEDULE_SYSTEM_PROMPT = f"""You are a scheduling assistant that creates an \
actual inspection schedule for police stations, grounded in the rules found \
in the circular excerpts provided below, and using ONLY the real police \
station names given to you under STATIONS — never invent, omit, or rename \
any station from that list. Rules:
1. Read the excerpts for any documented requirements about inspection \
frequency (e.g. at least once a year), duration (e.g. minimum days for \
regular vs important stations), submission deadlines (e.g. an annual \
programme due by a specific date), and how work should be divided (e.g. \
shared between the S.P. and Additional S.P.). If the excerpts do not \
specify a particular rule, say so plainly in a note rather than inventing \
a plausible-sounding number.
2. {_CITATION_RULE}
3. Using ONLY the station list given to you under STATIONS, distribute \
EVERY station across the months of the coming year so each is scheduled \
consistent with the frequency rule found in the excerpts (default to once \
per year if no explicit frequency rule is found, and say so). Spread \
stations roughly evenly across months unless the excerpts give another \
basis for prioritizing certain months. If a station is marked "important" \
in the list you were given, apply whatever extra duration the excerpts \
specify for important stations.
4. Output the schedule as a Markdown table with columns: Month | Station | \
Duration | Notes. The final table must include every station from the \
STATIONS list exactly once (or more than once only if the frequency rule \
found in the excerpts requires more than one inspection per year, in \
which case each occurrence is its own row) — never add a station that \
was not given to you, never drop one that was.
5. Do not add a closing summary after the table; end after the last row.
"""


def build_schedule_context(stations: list[str], details: str = "",
                           history: list[dict] | None = None):
    """Returns (prompt, references, hits). Grounds an actual month-by-month
    inspection schedule in retrieved circular rules, using only the real
    station names provided — the station roster itself is never something
    a circular would contain, so it must come from the caller, not be
    invented by the model."""
    query = f"inspection schedule frequency duration months programme {details}".strip()
    hits = retrieve(query)

    doc_ids, context_parts = [], []
    for h in hits:
        did = h["meta"]["doc_id"]
        if did not in doc_ids:
            doc_ids.append(did)
        issuer = _short_issuer(h['meta'].get('issuer', ''))
        year = h['meta'].get('year', 'n.d.')
        excerpt = h["text"][:config.PROMPT_CHUNK_CHARS]
        context_parts.append(
            f">> EXCERPT_METADATA issuer={issuer} | year={year} | "
            f"file={h['meta']['filename']} << (never print this metadata "
            f"line itself; only use its values to build a citation)\n{excerpt}"
        )

    excerpts_block = ("\n\n---\n\n".join(context_parts) if context_parts
                      else "(No relevant circulars were found in the repository — "
                           "note this plainly and use a default of once-per-year, "
                           "evenly distributed, with no specified duration.)")
    stations_block = "\n".join(f"- {s}" for s in stations) if stations else "(none provided)"
    extra = f"\nAdditional context from the requester: {details}" if details.strip() else ""

    prompt = (SCHEDULE_SYSTEM_PROMPT
              + "\n\n=== EXCERPTS ===\n" + excerpts_block
              + f"\n\n=== STATIONS ===\n{stations_block}{extra}"
              + "\n\n=== SCHEDULE ===\n")

    refs = [build_apa_reference(store.get_document(d))
            for d in doc_ids if store.get_document(d)]
    return prompt, refs, hits


def build_template_context(doc_type: str, details: str = "",
                           history: list[dict] | None = None):
    """Returns (prompt, references, hits). Unlike build_context, an empty
    repository does not short-circuit — the model still drafts a generic
    template but is instructed to flag that clearly."""
    query = f"{doc_type} format प्रपत्र नमुना {details}".strip()
    hits = retrieve(query)

    doc_ids, context_parts = [], []
    for h in hits:
        did = h["meta"]["doc_id"]
        if did not in doc_ids:
            doc_ids.append(did)
        issuer = _short_issuer(h['meta'].get('issuer', ''))
        year = h['meta'].get('year', 'n.d.')
        excerpt = h["text"][:config.PROMPT_CHUNK_CHARS]
        # Deliberately NOT shaped like a citation (no parentheses, no
        # "Source" word, no em-dash) so a weak model is less likely to
        # copy this label wholesale into its answer instead of extracting
        # just the citation values from it.
        context_parts.append(
            f">> EXCERPT_METADATA issuer={issuer} | year={year} | "
            f"file={h['meta']['filename']} << (never print this metadata "
            f"line itself; only use its values to build a citation)\n{excerpt}"
        )

    extra = f"\nAdditional context from the requester: {details}" if details.strip() else ""
    excerpts_block = ("\n\n---\n\n".join(context_parts) if context_parts
                      else "(No relevant circulars were found in the repository.)")

    prompt = (TEMPLATE_SYSTEM_PROMPT
              + "\n\n=== EXCERPTS ===\n" + excerpts_block
              + f"\n\n=== DOCUMENT TYPE REQUESTED ===\n{doc_type}{extra}"
              + "\n\n=== TEMPLATE ===\n")

    refs = [build_apa_reference(store.get_document(d))
            for d in doc_ids if store.get_document(d)]
    return prompt, refs, hits


def answer(question: str, history: list[dict] | None = None) -> dict:
    prompt, refs, hits = build_context(question, history)
    if prompt is None:
        return {"answer": EMPTY_MSG, "references": [], "hits": []}
    english = ask_phi3(prompt)
    result = {"answer": english, "references": refs, "hits": hits}
    if config.TRANSLATION_ENABLED and translation.is_devanagari_question(question):
        translated = translation.translate_markdown_to_marathi(english)
        if translated is not None:
            result["answer"] = translated
            result["answer_en"] = english
    return result


def stream_phi3(prompt: str):
    """Yield answer text piece by piece. No request in the chain ever idles
    longer than the gap between tokens, so proxy/browser timeouts don't fire."""
    try:
        with requests.post(
            config.OLLAMA_URL,
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": True,
                  "keep_alive": config.KEEP_ALIVE, "options": _ollama_options()},
            stream=True, timeout=(10, config.LLM_TIMEOUT),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]
                if data.get("done"):
                    return
    except requests.ConnectionError:
        yield (f"⚠️ Could not reach Ollama at {config.OLLAMA_URL}. "
               f"Run `ollama serve` and `ollama pull {config.LLM_MODEL}`.")
    except Exception as e:
        yield f"⚠️ LLM error: {e}"
