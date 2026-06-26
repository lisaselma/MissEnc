"""Table cell normalization + missingness detection (structural cleaning).

This module owns TWO concerns, both value-preserving:

  1. Structural cleaning (headers + cell-text form). The legacy helpers
     `_normalize`, `_normalize_columns`, `_split_camel_snake`,
     `_expand_abbreviations`, `ABBREVIATIONS` derive from cell 3 of
     cell_level/prepare_table.ipynb. They normalize unicode/whitespace,
     split camel/snake case, expand abbreviations and case-fold. They
     NEVER substitute disguised tokens and NEVER replace missing values.
     Casing is handled by `_downcase_capslock`: ALL-CAPS tokens are
     lowercased for consistency, except genuine acronyms (length <= 3 or
     in `ACRONYM_ALLOWLIST`), which keep their casing.

  2. Missingness DETECTION (not substitution): `DISGUISED_LITERALS`,
     `DISGUISED_REGEX_LABELED`, `NUMERIC_SENTINELS`, `DATE_SENTINELS`,
     `is_missing`. Detection is the single source of truth consumed by
     the sampler to *label* cells; it never rewrites a value.

Masking / token routing (the `[?]` / `[EMPTY]` substitution and the
`EMPTY_SENTINEL`/`canonicalize_missing` machinery) does NOT live here.
By the locked step order it happens only in `02_input_prep.py`, after
sampling. Keep this module free of any token substitution.
"""

import re
import unicodedata

import pandas as pd


# Common abbreviation expansions for tabular metadata. Whole-word,
# case-insensitive (see `_expand_abbreviations`); growing the dict has no
# perf cost. Domain entries cover the clinical/biomedical/genomic corpus.
# Ambiguous abbreviations are resolved per-corpus (see CLEANUP_SPEC §1.2):
# 'org' -> organization (administrative reporter tables), 'sr' -> sinus
# rhythm (cardiology subgroups), 'dm' -> dry matter (nutrition). Device /
# study-specific codes (cnap, tline, ncat) and the plural artifact 's'
# are deliberately NOT mapped.
ABBREVIATIONS = {
    "avg": "average",
    "len": "length",
    "num": "number",
    "id": "identifier",
    "qty": "quantity",
    "amt": "amount",
    "min": "minimum",
    "max": "maximum",
    "std": "standard",
    "dev": "deviation",
    "freq": "frequency",
    "src": "source",
    "dst": "destination",
    "msg": "message",
    "geo": "geographic",
    "img": "image",
    "vid": "video",
    "doc": "document",
    "ref": "reference",
    "loc": "location",
    "addr": "address",
    "desc": "description",
    "config": "configuration",
    "info": "information",
    "lib": "library",
    "lib_name": "library name",
    # --- domain additions (CLEANUP_SPEC §1.2) ---
    # clinical vitals / measures
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "bp": "blood pressure",
    "hr": "heart rate",
    "bmi": "body mass index",
    "egfr": "estimated glomerular filtration rate",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    # conditions
    "afib": "atrial fibrillation",
    "sr": "sinus rhythm",
    "copd": "chronic obstructive pulmonary disease",
    "htn": "hypertension",
    "ckd": "chronic kidney disease",
    # care / admin
    "icu": "intensive care unit",
    "los": "length of stay",
    "hosp": "hospital",
    "org": "organization",
    "pmi": "post mortem interval",
    # genomics / lab
    "rin": "rna integrity number",
    "sra": "sequence read archive",
    "pcr": "polymerase chain reaction",
    "rna": "ribonucleic acid",
    "dna": "deoxyribonucleic acid",
    # units / misc
    "mg": "milligram",
    "dm": "dry matter",
    "gdp": "gross domestic product",
    # "exp": "experiment",
}


# Genuine multi-letter acronyms (>3 chars) that should KEEP their casing
# when `_normalize` downcases ALL-CAPS cell tokens. Short all-caps tokens
# (length <= 3, e.g. RNA, USA, II) are treated as acronyms automatically;
# this allowlist covers the longer ones that would otherwise be lowercased.
# Stored uppercase; extend with corpus-specific acronyms as needed.
ACRONYM_ALLOWLIST: frozenset[str] = frozenset({
    "ELISA", "FASTA", "FASTQ", "BLAST", "GWAS",
    "NCBI", "LOINC", "SNOMED", "HIPAA",
})


def _split_camel_snake(s: str) -> str:
    """Convert CamelCase / snake_case / kebab-case to space-separated words."""
    s = re.sub(r"[_\-]+", " ", s)                      # snake / kebab -> space
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)         # camelCase -> camel Case
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)   # ABCWord -> ABC Word
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _expand_abbreviations(s: str) -> str:
    """Replace known abbreviations with full words (case-insensitive, whole-word)."""
    def repl(m):
        word = m.group(0).lower()
        return ABBREVIATIONS.get(word, m.group(0))
    return re.sub(r"\b\w+\b", repl, s)


def _downcase_capslock(s: str) -> str:
    """Lowercase tokens while preserving genuine acronyms.

    Replaces the blanket ``.lower()`` so capslock cell values
    (``GLUCAGON``, ``TAPEWORM INFECTIONS``, ``INSULIN INDUCED
    HYPOGLYCEMIA``) normalize consistently with their mixed/lower-case
    counterparts. A token is kept as-is only when it is ALL-CAPS AND
    either at most 3 characters (``RNA``, ``USA``, ``II``) or listed in
    ``ACRONYM_ALLOWLIST``; every other token is lowercased, so normal
    words behave exactly as before.
    """
    def repl(m):
        tok = m.group(0)
        if tok.isupper() and (len(tok) <= 3 or tok in ACRONYM_ALLOWLIST):
            return tok
        return tok.lower()
    return re.sub(r"[A-Za-z0-9]+", repl, s)


def _normalize(v: str) -> str:
    s = unicodedata.normalize("NFKD", str(v).strip()).encode("ascii", "ignore").decode()
    s = re.sub(r"\s*\|\s*", " ", s)
    s = re.sub(r"\b([A-Za-z])\.\s*([A-Za-z])\.", r"\1\2", s)
    s = _split_camel_snake(s)
    s = _expand_abbreviations(s)
    return _downcase_capslock(s)


def _normalize_columns(df):
    df = df.copy()
    df.columns = [_normalize(c) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Missingness DETECTION (§6 of IMPLEMENTATION_SPEC.md).
# Detection only: these never substitute a value. Substitution (routing to
# [EMPTY]/[?]) lives in 02_input_prep.py.
# ---------------------------------------------------------------------------

DISGUISED_LITERALS: frozenset[str] = frozenset({
    # database / dump conventions
    "null", "nil", "none", "nan", r"\n",
    # human-typed "no value"
    "n/a", "na", "n.a.", "n-a",
    "n/d", "nd",
    "unknown", "unk", "undisclosed", "undetermined", "unspecified",
    "unidentified", "unreported", "unavailable",
    "not available", "not applicable", "not specified", "not reported",
    "not given", "not assigned", "not provided", "not stated", "not known",
    "not collected", "not recorded", "not determined", "not listed",
    "not measured", "not tested", "not disclosed", "not found",
    # wrapped / compound forms (corpus scan: ep_research_data, etc.)
    "-- not applicable --", "coded data not available", "inapplicable",
    "missing", "missing value", "missing data", "value missing", "data missing",
    "no data", "nodata", "no value", "no info", "no information",
    "no entry", "no answer", "no response", "no record",
    "don't know", "dont know", "dont-know", "do not know", "did not respond",
    "tbd", "tba", "pending", "redacted", "withheld", "confidential",
    "refused", "dk", "rf", "dk/rf",
    # placeholders
    "?", "??", "???",
    "-", "--", "---", "—", "–", "_", "__", "___",
    ".", "..", "...", "…",
    "*", "**",
    "x", "xx", "xxx",
})


# Labelled regex patterns. The label flows into the sampler's occurrences
# audit (`matched_pattern` column) so the audit shows which family
# triggered; the unlabelled tuple below is what `is_missing()` consults.
DISGUISED_REGEX_LABELED: tuple[tuple[str, re.Pattern], ...] = (
    ("not_<verb>", re.compile(
        r"^not[\s_\-]+(available|applicable|specified|reported|given|assigned|"
        r"provided|stated|known|collected|recorded|determined|disclosed|"
        r"listed|measured|tested|found)\b"
    )),
    ("dash_wrapped_not_applicable", re.compile(
        r"^--\s*not applicable\s*--$"
    )),
    ("not_available_phrase", re.compile(
        r"^(?:[\w][\w\s\-]{0,30} )?not[\s_\-]+available$"
    )),
    ("<value>_missing", re.compile(
        r"^(value|data|info(?:rmation)?)[\s_\-]+missing\b"
    )),
    ("missing_<noun>", re.compile(
        r"^missing(?:[\s_\-]+(?:value|data|info|entry))?$"
    )),
    ("no_<noun>", re.compile(
        r"^no[\s_\-]+(data|value|info(?:rmation)?|entry|answer|response|record)\b"
    )),
    ("do(es)_not_<verb>", re.compile(
        r"^(?:do(?:es)?[\s_\-]+not|did[\s_\-]+not|don'?t|didn'?t)"
        r"[\s_\-]+(?:know|respond|answer|apply)\b"
    )),
    ("un<adj>", re.compile(
        r"^un(known|disclosed|determined|specified|identified|reported|available)\b"
    )),
    # Free-text values that OPEN with the standalone word "unknown" express
    # missingness/uncertainty, e.g. "unknown if a CR was achieved or was
    # lost to follow up", "unknown method". Start-anchored + \b so it never
    # fires on mid-sentence "... is unknown" or fused tokens like
    # "unknownhematopoeitic"; DOTALL so the trailing text may span newlines.
    ("unknown_<text>", re.compile(r"^unknown\b.*", re.DOTALL)),
    ("n/a_variant", re.compile(r"^n[/\.\-]?a\.?$")),
    ("n/d_variant", re.compile(r"^n[/\.\-]?d\.?$")),
    ("punctuation_only", re.compile(r"^[\?\-\.\*_\—–…]+$")),
    ("whitespace_only", re.compile(r"^\s+$")),
)
DISGUISED_REGEX: tuple[re.Pattern, ...] = tuple(p for _, p in DISGUISED_REGEX_LABELED)


NUMERIC_SENTINELS: frozenset[float] = frozenset({
    -1.0, -9.0, -99.0, -999.0, -9999.0, -99999.0,
    9999.0, 99999.0, 999999.0, 99999999.0,
})


# Date placeholders commonly used for "no real date known". Not used by
# `is_missing()` (target-column sampling stays strict); kept here as the
# single source of truth so the sampler doesn't fork its own copy.
DATE_SENTINELS: frozenset[str] = frozenset({
    "1900-01-01", "1899-12-30", "1970-01-01", "9999-12-31", "0000-00-00",
})


def is_missing(value) -> bool:
    """Global missingness predicate.

    Returns True iff:
      - value is NaN / None / pd.NA, OR
      - value is the empty string (after .strip()), OR
      - value (as a string, lowercased, stripped) is in DISGUISED_LITERALS, OR
      - value matches one of DISGUISED_REGEX (fullmatch), OR
      - value is a numeric type whose float() is in NUMERIC_SENTINELS.

    Detection is anchored full-match against literals/regexes/numeric
    sentinels, so long free-text never false-positives (a 200-char note
    cannot fullmatch a placeholder pattern or equal a literal); no length
    guard is needed.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    s = str(value).strip()
    if not s:
        return True

    # Numeric sentinel check: works for both numeric types and string
    # representations like "-999", "9999.0".
    try:
        f = float(s)
        if f in NUMERIC_SENTINELS:
            return True
    except (TypeError, ValueError):
        pass

    s_low = s.lower()
    if s_low in DISGUISED_LITERALS:
        return True
    return any(p.fullmatch(s_low) for p in DISGUISED_REGEX)


def format_cell_value(value) -> str:
    """Render a single present cell value for MiniLM serialization.

    Integers stored as floats (``42.0``) print without the trailing
    ``.0``; everything else is ``str()``-ed. Missingness is decided by the
    caller via ``is_missing`` (so disguised tokens map to [EMPTY]); this
    helper only handles the *present* rendering.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Structural cleaning only: normalize headers + cell-text form.

    Value-preserving by contract (CLEANUP_SPEC §1.1):
      - headers are normalized via `_normalize_columns`,
      - non-null cells are normalized via `_normalize` (unicode/whitespace,
        camel/snake split, abbreviation expansion, acronym-aware casing),
      - NaN/None cells are left untouched (no NaN introduced or removed),
      - no disguised token is substituted and nothing is routed to
        `[EMPTY]`/`[?]`.

    A cleaned table is semantically identical to its source; `is_missing`
    produces the same labels whether the sampler runs on `datasets/` or on
    a `datasets_clean/` produced here (detection is form-tolerant). The raw
    input is never mutated.
    """
    out = df.copy()

    def _norm_cell(v):
        try:
            if v is None or pd.isna(v):
                return v
        except (TypeError, ValueError):
            pass
        return _normalize(v)

    for col in out.columns:
        out[col] = out[col].map(_norm_cell)
    out = _normalize_columns(out)
    return out


def _materialize_clean_dir(in_dir, out_dir) -> int:
    """Clean every parquet under `in_dir` 1:1 into `out_dir`. Returns the
    number of files written. Never touches `in_dir`.

    Duplicate columns are left as-is (never de-duplicated). A table whose
    normalized headers collide cannot be written to Parquet, so it is
    skipped with a warning and the run continues.
    """
    from pathlib import Path

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.parquet"))
    n = 0
    n_skipped = 0
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            cleaned = clean_table(df)
            cleaned.to_parquet(out_dir / fp.name, index=False)
            n += 1
        except Exception as e:  # pragma: no cover - per-file guard
            n_skipped += 1
            print(f"[clean] SKIP {fp.name}: {type(e).__name__}: {e}")
    print(f"[clean] cleaned {n}/{len(files)} parquet files -> {out_dir} "
          f"({n_skipped} skipped)")
    return n


def main(argv=None):
    import argparse
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Standalone structural cleaning pass: datasets/ -> "
        "datasets_clean/ (value-preserving; no masking). Optional; the "
        "pipeline also cleans lazily in 02_input_prep.py.",
    )
    p.add_argument("--in-dir", type=Path, default=repo_root / "datasets")
    p.add_argument("--out-dir", type=Path, default=repo_root / "datasets_clean")
    args = p.parse_args(argv)
    _materialize_clean_dir(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
