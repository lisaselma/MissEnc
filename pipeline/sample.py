"""Unified sampler + disguised-missing detector (merges old 00a + 00b).

One self-contained entry point: given a flat `datasets/*.parquet` corpus,
produce `samples/sample_manifest.parquet` with FIRST-CLASS disguised
detection computed INLINE — no scan-then-join across two files, no CSV
dependency, no key-mismatch risk.

    python pipeline/sample.py --datasets datasets/ --out samples/

Pipeline (single pass, no ordering dependency on any prior step):
    scan -> classify -> eligibility -> random pick -> balanced draw
         -> co-missing -> manifest [-> optional occurrences audit]

The manifest is the standalone sanity-check artifact, inspectable before
any tokenization / embedding.

Single source of truth
----------------------
ALL disguised-detection knowledge lives in `01_table_cleaning.py` and is
imported, never forked:
  DISGUISED_LITERALS, DISGUISED_REGEX_LABELED, NUMERIC_SENTINELS,
  DATE_SENTINELS, is_missing().
The dtype-aware split (numeric vs object columns) is preserved via the
`classify_numeric` / `classify_string` helpers below, which consult those
same constants.

One unified missingness definition everywhere
---------------------------------------------
Per cell: `nan` (pd.isna) | `disguised` (is_missing & not nan; label
recovered via classify_*) | `present`. `y = 1` for nan OR disguised. This
one mask drives eligibility, the 5+5 balanced draw, and co-missing
detection. Disguised tokens count as MISSING for the balance: the
"missing" pool is `nan ∪ disguised`, the "present" pool is strictly
present.

Behaviour kept from 00a
-----------------------
Column eligibility (internal/constant/unique-ID/free-text/aggregate
screens), pure-random no-priority column pick, append-only / never-reuse
-a-row machinery, run-indexed seeds, 5+5 balance, co-missing detection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import (
    REPO_ROOT,
    git_sha,
    load_sibling_module,
    now_iso,
    sha256,
)

# Frozen detection knowledge — single source of truth.
clean = load_sibling_module("01_table_cleaning.py", alias="_cleaning_sample")
is_missing = clean.is_missing
_normalize = clean._normalize
DISGUISED_LITERALS = clean.DISGUISED_LITERALS
DISGUISED_REGEX_LABELED = clean.DISGUISED_REGEX_LABELED
NUMERIC_SENTINELS = clean.NUMERIC_SENTINELS
DATE_SENTINELS = clean.DATE_SENTINELS


# ---------------------------------------------------------------------------
# Inline disguised classification (ported from 00b; uses 01's constants).
# Returns (kind, label): kind in {literal, regex, date_sentinel,
# numeric_sentinel} or None; label feeds special_token / matched_pattern.
# ---------------------------------------------------------------------------
def classify_string(value) -> tuple[str | None, str | None]:
    """Classify an OBJECT-column cell as disguised-missing, else (None, None)."""
    if value is None:
        return None, None
    s = str(value).strip()
    if not s:
        return None, None
    s_low = s.lower()
    if s_low in DISGUISED_LITERALS:
        return "literal", s_low
    for label, pat in DISGUISED_REGEX_LABELED:
        if pat.fullmatch(s_low):
            return "regex", label
    if s_low in DATE_SENTINELS:
        return "date_sentinel", s_low
    return None, None


def classify_numeric(value) -> tuple[str | None, str | None]:
    """Classify a NUMERIC-column cell as a numeric sentinel, else (None, None)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None, None
    if pd.isna(f):
        return None, None
    if f in NUMERIC_SENTINELS:
        return "numeric_sentinel", str(int(f) if f.is_integer() else f)
    return None, None


def classify_cell(value, is_numeric_col: bool) -> tuple[str, str | None, str]:
    """Per-cell classification: (missing_kind, special_token, original_value).

    missing_kind in {present, nan, disguised}. For disguised cells the
    matched literal/label is the `special_token` and the raw cell is the
    `original_value`. `is_missing` is the authoritative missing predicate;
    classify_* enriches the label when the dtype-aware scan recognises it.
    """
    try:
        if pd.isna(value):
            return "nan", None, ""
    except (TypeError, ValueError):
        pass

    kind, label = classify_numeric(value) if is_numeric_col else classify_string(value)
    if kind is not None:
        return "disguised", label, str(value)
    # is_missing catches cases classify_* skips (e.g. "" or a numeric
    # sentinel stored as a string in an object column) — still disguised,
    # just without a clean label.
    if is_missing(value):
        # Empty / whitespace-only strings get a readable label instead of
        # an unreadable "" in special_token.
        if isinstance(value, str) and value.strip() == "":
            return "disguised", "empty string", str(value)
        return "disguised", None, str(value)
    return "present", None, str(value)


def _enriched_is_missing(series: pd.Series) -> pd.Series:
    """Boolean missing mask (NaN OR disguised) aligned with the index."""
    return series.map(is_missing).astype(bool)


# Collapse trivial column-name variants into one canonical column_type.
# Currently only the 'age 2' / 'age 3' / ... -> 'age' family (gender/sex
# stay DISTINCT by request). Add more rules here if other split-numbered
# variants appear.


# Versioned label rename map (reproducibility). Applied to the NORMALIZED
# column name to produce the canonical `column_type`. This is a LABEL rename
# only -- it never merges columns (provenance stays via file + target_col_idx
# + row_idx) and never alters datasets/ on disk. Keys are post-_normalize
# strings; values are the canonical label. Edit here to change labels
# reproducibly across every run.
RENAME_MAP: dict[str, str] = {
    "duration s": "duration",
    "binfo followers": "information followers",
    "binfo website": "information website",
    "location 2": "location",
    "location 3": "location",
    "index level 10": "index level",
    "index level 11": "index level",
    "index level 12": "index level",
    "index level 14": "index level",
    "total.tested.today": "total tested today",
    "total.tested": "total tested",
    "total.death.today": "total death today",
    "total.death": "total death",
    "total.case": "total case",
    "positivity.rate": "positivity rate",
    "growth rate 20112016": "growth rate between 2011 and 2016",
    "age 2": "age",
    "age 3": "age",
    "city 2": "city",
    "gender 2": "gender",
    "gender 3": "gender",
    "n": "number",
    "collection date (1)": "collection date",
    "temp high": "temperature high",
    "tissue site 2": "tissue site",
    "b cell type": "cell type",
    "vonsattel grade s": "vonsattel grade",
    "h v cortical score s": "cortical score",
    "clinical stage 2": "clinical stage",
    "other activity method comment.1": "other activity method comment",
    "energy content k j) (original)": "original energy content",
    "reviewer comments.1 (1)": "reviewer comments",
}


def _column_type(name: str) -> str:
    """Canonical column label: normalize, then apply the versioned RENAME_MAP.

    A rename only -- columns are NEVER merged. Two distinct source columns
    keep distinct provenance (file + target_col_idx + row_idx); only their
    human-readable label may coincide after renaming (e.g. age + age 2 -> age).
    Deterministic and versioned so every run reproduces the same labels.
    """
    n = _normalize(name)
    return RENAME_MAP.get(n, n)


# Default soft-priority families (matched on the normalized column name).
# Columns whose normalized name matches are moved to the FRONT of the
# random pick order, so they get sampled before the per-file cap bites.
# gender and sex are listed separately and remain distinct column_types.
DEFAULT_PRIORITY_REGEX: str = (
    r"\bage\b|\bsex\b|\bgender\b|\brace\b|ethnic|"
    r"\bcountry\b|\bcity\b|\blocation\b|\bregion\b|\bplace\b|geographic|"
    r"\bdate\b|\bdisease\b|pathology|diagnosis|\borganism\b|\bspecies\b"
)


# ---------------------------------------------------------------------------
# Column eligibility (kept from 00a).
# ---------------------------------------------------------------------------
AGGREGATE_CELL_RE: re.Pattern = re.compile(
    r"^\s*(?:[^:;,]+:\s*<?\s*\d+(?:\.\d+)?\s*%?\s*[;,]?\s*){2,}$"
)
INTERNAL_COL_RE: re.Pattern = re.compile(
    r"^(?:__index_level_\d+__|__pandas[\w_]*|Unnamed:\s*\d+|"
    r"unnamed(?::\s*|\s+column\s+)\d+)$",
    re.IGNORECASE,
)

# Hard block-list of specific (file, raw column name) pairs that are known
# bad parses (header row mangled into data). These are skipped regardless
# of content so they are never sampled again.
BLOCKED_FILE_COLUMNS: set[tuple[str, str]] = {
    ("003_backend_repertoire.parquet", "unnamed column 17"),
    ("135_test_state_data_basic.parquet", "unnamed: 2"),
    ("063_federal_agencies_1.parquet", "ame"),
    ("063_federal_agencies_1.parquet", "cy"),
    ("085_20170111_articles.parquet", "aims"),
    ("085_20170111_articles.parquet", "to"),
    ("085_20170111_articles.parquet", "to.1"),
    ("094_20170109_articles.parquet", "to"),
    ("094_20170109_articles.parquet", "with"),
}


def _column_stats(series: pd.Series) -> dict:
    miss_mask = _enriched_is_missing(series)
    n_m = int(miss_mask.sum())
    n_p = int((~miss_mask).sum())
    n_d = int((miss_mask & ~series.isna()).sum())
    if n_p == 0:
        return {
            "n_missing": n_m, "n_present": 0, "n_disguised": n_d,
            "n_unique_present": 0, "uniq_frac": 0.0,
            "mean_present_len": 0.0, "agg_frac": 0.0, "miss_mask": miss_mask,
        }
    present = series[~miss_mask]
    n_u = int(present.nunique(dropna=False))
    present_str = present.astype("string").fillna("")
    n_agg = int(present_str.str.match(AGGREGATE_CELL_RE).sum())
    return {
        "n_missing": n_m, "n_present": n_p, "n_disguised": n_d,
        "n_unique_present": n_u, "uniq_frac": n_u / n_p,
        "mean_present_len": float(present_str.str.len().mean()),
        "agg_frac": n_agg / n_p, "miss_mask": miss_mask,
    }


def _eligibility_reason(
    stats: dict, column_name: str, min_n_per_class: int,
) -> str | None:
    # NOTE: unique-ID / free-text / aggregate-column filters intentionally
    # removed (not relevant to this research). Only structural screens
    # remain: internal/index columns, all-constant columns, and the
    # missing/present minimum needed for a balanced 5/5 draw.
    if INTERNAL_COL_RE.match(str(column_name)):
        return "internal_column"
    if stats["n_missing"] < min_n_per_class:
        return "lt_n_missing"
    if stats["n_present"] < min_n_per_class:
        return "lt_n_present"
    if stats["n_unique_present"] <= 1:
        return "constant"
    return None


def _scan_columns(
    df: pd.DataFrame, *, min_n_per_class: int, file_basename: str = "",
) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    eligible: list[dict] = []
    for col in df.columns:
        stats = _column_stats(df[col])
        if (file_basename, str(col)) in BLOCKED_FILE_COLUMNS:
            reason = "blocked_bad_parse"
        else:
            reason = _eligibility_reason(
                stats, column_name=col, min_n_per_class=min_n_per_class,
            )
        row = {
            "column": str(col),
            "n_missing": stats["n_missing"], "n_present": stats["n_present"],
            "n_disguised": stats["n_disguised"],
            "uniq_frac": round(stats["uniq_frac"], 4),
            "mean_present_len": round(stats["mean_present_len"], 2),
            "agg_frac": round(stats["agg_frac"], 4),
            "eligibility_reason": reason, "miss_mask": stats["miss_mask"],
        }
        candidates.append(row)
        if reason is None:
            eligible.append(row)
    return eligible, candidates


def _detect_co_missing_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Context columns perfectly co-missing with the target under the
    enriched mask (P(t|c)==1 and P(c|t)==1). Stored once per (file,target)
    so the co_missing_drop ablation stays a cheap toggle downstream."""
    t_mask = _enriched_is_missing(df[target_col])
    n_t = int(t_mask.sum())
    if n_t == 0:
        return []
    dropped: list[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        c_mask = _enriched_is_missing(df[c])
        n_c = int(c_mask.sum())
        if n_c == 0:
            continue
        inter = int((t_mask & c_mask).sum())
        if (inter / n_c) == 1.0 and (inter / n_t) == 1.0:
            dropped.append(str(c))
    return dropped


# ---------------------------------------------------------------------------
# Append-only state + run-indexed seeds (kept from 00a).
# ---------------------------------------------------------------------------
def _file_seed(base_seed: int, file_basename: str, run_index: int = 0) -> int:
    h = hashlib.blake2b(file_basename.encode("utf-8"), digest_size=8).digest()
    return ((base_seed * 1000003) ^ int.from_bytes(h, "big")) ^ int(run_index)


def _load_existing_manifest_state(
    path: Path,
) -> tuple[dict[str, set[int]], dict[str, int], int,
           dict[str, int], dict[tuple[str, str], int], dict[str, set[str]]]:
    """Returns:
      rows_by_file        : {file_basename -> set(row_idx)}  (never reuse a row)
      runidx_by_file      : {file_basename -> n distinct cols already sampled}
      next_sample_id      : max(sample_id)+1
      ctype_count         : {column_type -> n manifest rows}  (optional global cap)
      filecol_count       : {(file_basename, target_col) -> n manifest rows}  (per-file cap)
      cols_seen           : {file_basename -> set(target_col)} already sampled
    """
    rows_by_file: dict[str, set[int]] = {}
    runidx_by_file: dict[str, int] = {}
    ctype_count: dict[str, int] = {}
    filecol_count: dict[tuple[str, str], int] = {}
    cols_seen: dict[str, set[str]] = {}
    if not path.exists():
        return rows_by_file, runidx_by_file, 0, ctype_count, filecol_count, cols_seen
    df = pd.read_parquet(
        path,
        columns=["file_basename", "target_col", "row_idx", "sample_id", "column_type"],
    )
    if df.empty:
        return rows_by_file, runidx_by_file, 0, ctype_count, filecol_count, cols_seen
    for fb, tc, ri, ct in zip(
        df["file_basename"].astype(str), df["target_col"].astype(str),
        df["row_idx"].astype(int), df["column_type"].astype(str),
    ):
        rows_by_file.setdefault(fb, set()).add(int(ri))
        cols_seen.setdefault(fb, set()).add(tc)
        ctype_count[ct] = ctype_count.get(ct, 0) + 1
        filecol_count[(fb, tc)] = filecol_count.get((fb, tc), 0) + 1
    runidx_by_file = {fb: len(cols) for fb, cols in cols_seen.items()}
    return (rows_by_file, runidx_by_file, int(df["sample_id"].max()) + 1,
            ctype_count, filecol_count, cols_seen)


# ---------------------------------------------------------------------------
# Optional occurrences audit (diagnostic side output; nothing reads it).
# ---------------------------------------------------------------------------
def _scan_occurrences(df: pd.DataFrame, file_basename: str, table_id: str) -> list[dict]:
    """One record per disguised cell across ALL columns (dtype-aware)."""
    records: list[dict] = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        for row_idx, val in series.items():
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            kind, label = classify_numeric(val) if is_numeric else classify_string(val)
            if kind is None:
                continue
            records.append({
                "table_id": table_id, "file": file_basename, "column": str(col),
                "row_idx": int(row_idx) if isinstance(row_idx, (int, np.integer)) else row_idx,
                "original_value": str(val),
                "normalized_value": str(val).strip().lower(),
                "matched_pattern": label, "kind": kind,
            })
    return records


def _summarize_occurrences(occ: pd.DataFrame) -> pd.DataFrame:
    if occ.empty:
        return pd.DataFrame(columns=[
            "normalized_value", "kind", "matched_pattern",
            "n_occurrences", "n_columns", "n_files",
        ])
    return (
        occ.groupby(["normalized_value", "kind", "matched_pattern"], dropna=False)
           .agg(n_occurrences=("file", "size"),
                n_columns=("column", "nunique"),
                n_files=("file", "nunique"))
           .reset_index()
           .sort_values("n_occurrences", ascending=False)
           .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Retroactive cap enforcement on an existing manifest.
# ---------------------------------------------------------------------------
def prune_manifest_to_caps(
    manifest_path: Path,
    *,
    max_per_column_type: int | None = None,
    max_per_file_column: int = 20,
    random_state: int = 42,
) -> Path:
    """Delete rows from an existing manifest so it complies with the caps.

    Two caps, enforced in this order:
      1. per (file_basename, target_col) <= max_per_file_column
      2. per column_type (global)        <= max_per_column_type

    Within any over-cap group, rows are dropped to keep the missing/present
    balance as even as possible (drop from the majority class first), and
    selection is deterministic given random_state. Rows are never added.
    """
    df = pd.read_parquet(manifest_path)
    n0 = len(df)
    if df.empty:
        print("[prune] manifest empty; nothing to do")
        return manifest_path

    rng = np.random.default_rng(random_state)

    def _keep_balanced(sub: pd.DataFrame, k: int) -> pd.DataFrame:
        """Pick k rows from sub, keeping y-balance as even as possible."""
        if len(sub) <= k:
            return sub
        miss = sub[sub["is_missing"]]
        pres = sub[~sub["is_missing"]]
        n_miss = min(len(miss), (k + 1) // 2)
        n_pres = min(len(pres), k - n_miss)
        # if one class is short, backfill from the other
        n_miss = min(len(miss), k - n_pres)
        def _sample(g, n):
            if n <= 0:
                return g.iloc[0:0]
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
            return g.loc[idx]
        return pd.concat([_sample(miss, n_miss), _sample(pres, n_pres)])

    # ---- cap 1: per (file_basename, target_col) ----
    kept_parts = []
    for _, g in df.groupby(["file_basename", "target_col"], sort=False):
        kept_parts.append(_keep_balanced(g, max_per_file_column))
    df = pd.concat(kept_parts).sort_values("sample_id").reset_index(drop=True)
    n1 = len(df)

    # ---- cap 2: per column_type (global) -- OPTIONAL ----
    # By design, a column_type may appear across ANY number of datasets
    # (e.g. "age" in many files is desirable for cross-schema coverage).
    # Only applied when max_per_column_type is not None.
    if max_per_column_type is not None:
        kept_parts = []
        for ct, g in df.groupby("column_type", sort=False):
            if len(g) <= max_per_column_type:
                kept_parts.append(g)
                continue
            kept_parts.append(_keep_balanced(g, max_per_column_type))
        df = pd.concat(kept_parts).sort_values("sample_id").reset_index(drop=True)
    n2 = len(df)

    # ---- verify ----
    ct = df["column_type"].value_counts()
    fc = df.groupby(["file_basename", "target_col"]).size()
    if max_per_column_type is not None:
        assert ct.max() <= max_per_column_type, "column_type cap still violated"
    assert fc.max() <= max_per_file_column, "file_column cap still violated"

    tmp = manifest_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(manifest_path)
    print(f"[prune] {n0:,} -> after file/col cap {n1:,} -> after column_type cap {n2:,} rows  "
          f"(removed {n0 - n2:,})")
    print(f"[prune] max per column_type={int(ct.max())} (cap {max_per_column_type}), "
          f"max per (file,col)={int(fc.max())} (cap {max_per_file_column})")
    return manifest_path


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def build_sample_manifest(
    *,
    repo_root: Path,
    datasets_root: Path,
    out_dir: Path,
    n_per_class: int = 5,
    min_n_per_class: int | None = None,
    random_state: int = 42,
    n_target_types_per_table: int = 1,
    column_skip_if_already_present: bool = True,
    priority_column_types: str | None = DEFAULT_PRIORITY_REGEX,
    max_per_column_type: int | None = None,
    max_per_file_column: int = 20,
    limit_files: int | None = None,
    write_occurrences: bool = False,
) -> Path:
    """Scan + classify + sample in a single pass. Returns the manifest path.

    Auto-append: if a manifest already exists at
    `out_dir/sample_manifest.parquet`, its rows are excluded (never
    reused), sample_id continues from its max, and new rows are appended.
    """
    if random_state < 0:
        raise ValueError("random_state must be a non-negative int")
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    if n_target_types_per_table < 1:
        raise ValueError("n_target_types_per_table must be >= 1")
    if max_per_column_type is not None and max_per_column_type < 1:
        raise ValueError("max_per_column_type must be >= 1 or None (unlimited)")
    if max_per_file_column < 1:
        raise ValueError("max_per_file_column must be >= 1")
    if min_n_per_class is None:
        min_n_per_class = n_per_class
    if not (1 <= min_n_per_class <= n_per_class):
        raise ValueError(
            f"min_n_per_class ({min_n_per_class}) must be in [1, n_per_class={n_per_class}]"
        )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.parquet"

    (rows_by_file, runidx_by_file, next_sample_id,
     ctype_count, filecol_count, cols_seen) = _load_existing_manifest_state(manifest_path)
    if next_sample_id > 0:
        n_excluded = sum(len(v) for v in rows_by_file.values())
        print(f"[sample] auto-append: next_sample_id={next_sample_id:,}; excluding "
              f"{n_excluded:,} prior (file,row) pairs across {len(rows_by_file)} files")

    print(f"[sample] config: n_per_class={n_per_class}, min_n_per_class={min_n_per_class}, "
          f"n_target_types_per_table={n_target_types_per_table}, "
          f"column_skip_if_already_present={column_skip_if_already_present}, "
          f"max_per_column_type={max_per_column_type}, "
          f"max_per_file_column={max_per_file_column}, "
          f"random_state={random_state}, write_occurrences={write_occurrences}")

    priority_re = (
        re.compile(priority_column_types) if priority_column_types else None
    )
    if priority_re is not None:
        print(f"[sample] soft-priority regex active: {priority_column_types!r}")

    datasets_root = datasets_root.resolve()
    parquet_files = sorted(datasets_root.glob("*.parquet"))
    if limit_files is not None:
        parquet_files = parquet_files[:limit_files]
    print(f"[sample] {len(parquet_files)} parquet files under {datasets_root}")

    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    co_missing_rows: list[dict] = []
    occurrence_rows: list[dict] = []
    sample_id_counter = next_sample_id
    n_sampled_groups = 0
    n_skipped_groups = 0

    for i, fp in enumerate(parquet_files, 1):
        file_basename = fp.name
        table_id = fp.stem
        try:
            file_rel = str(fp.resolve().relative_to(repo_root))
        except ValueError:
            file_rel = str(fp.resolve())
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  [{i}/{len(parquet_files)}] SKIP {file_basename}: read error: {e}")
            summary_rows.append({
                "file": file_basename, "target_col": "", "column_type": "",
                "n_rows_total": 0, "n_missing_total": 0, "n_present_total": 0,
                "n_disguised_total": 0, "uniq_frac": None, "mean_present_len": None,
                "aggregate_frac": None, "sampled": False,
                "n_sampled_missing": 0, "n_sampled_present": 0,
                "skip_reason": "read_error", "co_missing_n_dropped": 0,
            })
            continue

        if write_occurrences:
            occurrence_rows.extend(_scan_occurrences(df, file_basename, table_id))

        run_index = runidx_by_file.get(file_basename, 0)
        file_seed = _file_seed(random_state, file_basename, run_index)

        eligible, candidates = _scan_columns(
            df, min_n_per_class=min_n_per_class, file_basename=file_basename,
        )

        # Uniformly-random column order (stable base sort, then shuffle).
        col_rng = np.random.default_rng(file_seed ^ 0xC01C0DE)
        order = sorted(eligible, key=lambda r: r["column"])
        col_rng.shuffle(order)
        # Soft priority: stable-partition matching columns to the front so
        # they are picked before the per-file cap bites. Order WITHIN each
        # group stays the random shuffle above (so ties remain unbiased).
        if priority_re is not None:
            order.sort(key=lambda r: 0 if priority_re.search(_normalize(r["column"])) else 1)

        used_rows: set[int] = set(rows_by_file.get(file_basename, set()))
        picked_columns: set[str] = set()
        already_cols: set[str] = (
            cols_seen.get(file_basename, set())
            if column_skip_if_already_present else set()
        )
        n_success = 0

        for cand in order:
            if n_success >= n_target_types_per_table:
                break
            target_col = cand["column"]
            # When enabled, never re-sample a column already in the manifest
            # for this file -- spreads coverage to NEW columns across reruns.
            if target_col in already_cols:
                continue
            miss_mask = cand["miss_mask"]
            is_numeric_col = pd.api.types.is_numeric_dtype(df[target_col])

            miss_idx = np.array([
                int(r) for r in df.index[miss_mask] if int(r) not in used_rows
            ])
            pres_idx = np.array([
                int(r) for r in df.index[~miss_mask] if int(r) not in used_rows
            ])
            actual_n = min(n_per_class, len(miss_idx), len(pres_idx))
            if actual_n < min_n_per_class:
                summary_rows.append({
                    "file": file_basename, "target_col": target_col,
                    "column_type": _column_type(target_col), "n_rows_total": len(df),
                    "n_missing_total": cand["n_missing"],
                    "n_present_total": cand["n_present"],
                    "n_disguised_total": cand["n_disguised"],
                    "uniq_frac": cand["uniq_frac"],
                    "mean_present_len": cand["mean_present_len"],
                    "aggregate_frac": cand["agg_frac"], "sampled": False,
                    "n_sampled_missing": 0, "n_sampled_present": 0,
                    "skip_reason": "insufficient_rows_after_exclusion",
                    "co_missing_n_dropped": 0,
                })
                n_skipped_groups += 1
                continue

            # ---- domination caps (counted in manifest rows; include prior runs) ----
            column_type = _column_type(target_col)
            n_would_add = 2 * actual_n
            ct_now = ctype_count.get(column_type, 0)
            fc_now = filecol_count.get((file_basename, target_col), 0)
            if max_per_column_type is not None and ct_now + n_would_add > max_per_column_type:
                summary_rows.append({
                    "file": file_basename, "target_col": target_col,
                    "column_type": column_type, "n_rows_total": len(df),
                    "n_missing_total": cand["n_missing"],
                    "n_present_total": cand["n_present"],
                    "n_disguised_total": cand["n_disguised"],
                    "uniq_frac": cand["uniq_frac"],
                    "mean_present_len": cand["mean_present_len"],
                    "aggregate_frac": cand["agg_frac"], "sampled": False,
                    "n_sampled_missing": 0, "n_sampled_present": 0,
                    "skip_reason": "max_per_column_type_reached",
                    "co_missing_n_dropped": 0,
                })
                n_skipped_groups += 1
                continue
            if fc_now + n_would_add > max_per_file_column:
                summary_rows.append({
                    "file": file_basename, "target_col": target_col,
                    "column_type": column_type, "n_rows_total": len(df),
                    "n_missing_total": cand["n_missing"],
                    "n_present_total": cand["n_present"],
                    "n_disguised_total": cand["n_disguised"],
                    "uniq_frac": cand["uniq_frac"],
                    "mean_present_len": cand["mean_present_len"],
                    "aggregate_frac": cand["agg_frac"], "sampled": False,
                    "n_sampled_missing": 0, "n_sampled_present": 0,
                    "skip_reason": "max_per_file_column_reached",
                    "co_missing_n_dropped": 0,
                })
                n_skipped_groups += 1
                continue

            col_hash = int.from_bytes(
                hashlib.blake2b(str(target_col).encode("utf-8"), digest_size=8).digest(), "big",
            )
            rng = np.random.default_rng((file_seed * 1000003) ^ col_hash)
            chosen_miss = rng.choice(miss_idx, size=actual_n, replace=False)
            chosen_pres = rng.choice(pres_idx, size=actual_n, replace=False)
            used_rows.update(int(x) for x in chosen_miss)
            used_rows.update(int(x) for x in chosen_pres)

            chosen = np.empty(2 * actual_n, dtype=chosen_miss.dtype)
            chosen[0::2] = chosen_miss
            chosen[1::2] = chosen_pres

            dropped_cols = _detect_co_missing_columns(df, target_col)
            target_col_idx = int(df.columns.get_loc(target_col))

            for row_idx in chosen:
                row_idx_int = int(row_idx)
                val = df.at[row_idx_int, target_col]
                missing_kind, special_token, original_value = classify_cell(
                    val, is_numeric_col
                )
                is_miss = missing_kind != "present"
                manifest_rows.append({
                    "sample_id": sample_id_counter,
                    "file": file_rel,
                    "file_basename": file_basename, "table_id": table_id,
                    "column_type": column_type, "target_col": target_col,
                    "target_col_idx": target_col_idx, "row_idx": row_idx_int,
                    "y": int(is_miss), "is_missing": is_miss,
                    "missing_kind": missing_kind, "special_token": special_token,
                    "original_value": original_value,
                    "co_missing_columns": dropped_cols, "random_state": random_state,
                })
                sample_id_counter += 1

            summary_rows.append({
                "file": file_basename, "target_col": target_col,
                "column_type": column_type, "n_rows_total": len(df),
                "n_missing_total": cand["n_missing"], "n_present_total": cand["n_present"],
                "n_disguised_total": cand["n_disguised"], "uniq_frac": cand["uniq_frac"],
                "mean_present_len": cand["mean_present_len"],
                "aggregate_frac": cand["agg_frac"], "sampled": True,
                "n_sampled_missing": actual_n, "n_sampled_present": actual_n,
                "skip_reason": None, "co_missing_n_dropped": len(dropped_cols),
            })
            co_missing_rows.append({
                "file": file_basename, "target_col": target_col,
                "column_type": column_type,
                "co_missing_columns": json.dumps(dropped_cols, ensure_ascii=False),
                "n_co_missing_columns": len(dropped_cols), "total_cols": int(df.shape[1]),
            })
            picked_columns.add(target_col)
            ctype_count[column_type] = ct_now + n_would_add
            filecol_count[(file_basename, target_col)] = fc_now + n_would_add
            n_success += 1
            n_sampled_groups += 1

        # Audit rows for columns never picked this run.
        for cand in candidates:
            if cand["column"] in picked_columns:
                continue
            if cand["eligibility_reason"] is None:
                already = any(
                    s["file"] == file_basename and s["target_col"] == cand["column"]
                    for s in summary_rows
                )
                if already:
                    continue
                reason = "not_selected_this_run"
            else:
                reason = cand["eligibility_reason"]
            summary_rows.append({
                "file": file_basename, "target_col": cand["column"],
                "column_type": _column_type(cand["column"]), "n_rows_total": len(df),
                "n_missing_total": cand["n_missing"], "n_present_total": cand["n_present"],
                "n_disguised_total": cand["n_disguised"], "uniq_frac": cand["uniq_frac"],
                "mean_present_len": cand["mean_present_len"],
                "aggregate_frac": cand["agg_frac"], "sampled": False,
                "n_sampled_missing": 0, "n_sampled_present": 0,
                "skip_reason": reason, "co_missing_n_dropped": 0,
            })

        if i % 50 == 0 or i == len(parquet_files):
            print(f"  [{i}/{len(parquet_files)}] sampled groups (this run): "
                  f"{n_sampled_groups}, new manifest rows: {len(manifest_rows)}")

    # ---- assemble + persist (append to existing manifest) ----
    new_df = pd.DataFrame(manifest_rows)
    if not new_df.empty:
        new_df["co_missing_columns"] = new_df["co_missing_columns"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

    if manifest_path.exists():
        existing_df = pd.read_parquet(manifest_path)
        manifest_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if not new_df.empty else existing_df
        )
    else:
        manifest_df = new_df

    # Kept in-memory only (console skip-summary + invariants); not written.
    # The manifest is the sole output.
    summary_df = pd.DataFrame(summary_rows)

    tmp_path = manifest_path.with_suffix(".parquet.tmp")
    manifest_df.to_parquet(tmp_path, index=False)
    tmp_path.replace(manifest_path)

    # Optional disguised-occurrences audit (diagnostic only).
    occ_paths = None
    if write_occurrences:
        st_dir = out_dir / "special_tokens"
        st_dir.mkdir(parents=True, exist_ok=True)
        occ_df = pd.DataFrame(occurrence_rows)
        occ_summary = _summarize_occurrences(occ_df)
        occ_path = st_dir / "occurrences_special_tokens.csv"
        sum_path = st_dir / "summary_special_tokens.csv"
        occ_df.to_csv(occ_path, index=False)
        occ_summary.to_csv(sum_path, index=False)
        occ_paths = (occ_path, sum_path)
        print(f"[sample] wrote {len(occ_df):,} disguised occurrences -> {occ_path}")

    # Per-group invariants on the NEW rows.
    if not new_df.empty:
        for (f, tc), g in new_df.groupby(["file", "target_col"], sort=False):
            n_miss = int(g["is_missing"].sum())
            n_pres = int((~g["is_missing"]).sum())
            if n_miss != n_pres:
                print(f"[sample] INVARIANT VIOLATION ({f}, {tc}): "
                      f"miss={n_miss}, pres={n_pres}, expected balanced")
            if n_miss < min_n_per_class or n_miss > n_per_class:
                print(f"[sample] INVARIANT VIOLATION ({f}, {tc}): "
                      f"n_per_class={n_miss} outside [{min_n_per_class}, {n_per_class}]")


    print(f"[sample] wrote manifest -> {manifest_path}  "
          f"(+{len(new_df):,} new rows, {len(manifest_df):,} total, "
          f"{n_sampled_groups} groups this run, {n_skipped_groups} skipped)")
    return manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", "--datasets-root", dest="datasets", type=Path,
                   default=REPO_ROOT / "datasets",
                   help="Flat directory of *.parquet (default: repo/datasets)")
    p.add_argument("--out", "--out-dir", dest="out", type=Path,
                   default=REPO_ROOT / "samples",
                   help="Output directory (default: samples)")
    p.add_argument("--n-per-class", type=int, default=5,
                   help="Missing AND present rows per chosen (file, target_col).")
    p.add_argument("--min-n-per-class", type=int, default=None,
                   help="Lower bound per class after row-exclusion. Below it the "
                        "column is skipped (insufficient_rows_after_exclusion). "
                        "Default: n_per_class (strict 5/5).")
    p.add_argument("--n-target-types-per-table", type=int, default=1,
                   help="Random eligible columns to sample per file per run.")
    p.add_argument("--random-state", type=int, default=42,
                   help="Base seed; combined with per-file run_index so reruns "
                        "explore new columns/rows.")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Smoke test: only scan first N parquet files")
    p.add_argument("--no-column-skip-if-already-present", dest="column_skip_if_already_present",
                   action="store_false",
                   help="Allow re-sampling columns that already appear in the manifest "
                        "for a file. Default: skip them (spread to new columns).")
    p.set_defaults(column_skip_if_already_present=True)
    p.add_argument("--priority-column-types", type=str, default=DEFAULT_PRIORITY_REGEX,
                   help="Regex on the normalized column name. Matching columns are "
                        "soft-prioritized (picked before the per-file cap). Pass a "
                        "custom regex to override the default demographic set.")
    p.add_argument("--no-priority", dest="priority_column_types",
                   action="store_const", const=None,
                   help="Disable soft priority; use pure random column order.")
    p.add_argument("--max-per-column-type", type=int, default=None,
                   help="OPTIONAL global cap: a column_type may appear in at most this "
                        "many manifest rows total (across all datasets + prior runs). "
                        "Default None = UNLIMITED (a type like 'age' can span any number "
                        "of datasets; only the per-file cap prevents one file dominating).")
    p.add_argument("--max-per-file-column", type=int, default=20,
                   help="Per-dataset cap: a (file, column) may appear in at most this "
                        "many manifest rows total (across prior runs).")
    p.add_argument("--write-occurrences", action="store_true",
                   help="Also emit the diagnostic disguised-occurrences CSVs under "
                        "<out>/special_tokens/. Nothing downstream reads them.")
    p.add_argument("--prune-only", action="store_true",
                   help="Do not sample. Load the existing manifest at <out>/"
                        "sample_manifest.parquet and DELETE rows so it complies "
                        "with --max-per-column-type and --max-per-file-column.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.prune_only:
        prune_manifest_to_caps(
            (args.out / "sample_manifest.parquet").resolve(),
            max_per_column_type=args.max_per_column_type,
            max_per_file_column=args.max_per_file_column,
            random_state=args.random_state,
        )
        return
    build_sample_manifest(
        repo_root=REPO_ROOT,
        datasets_root=args.datasets,
        out_dir=args.out,
        n_per_class=args.n_per_class,
        min_n_per_class=args.min_n_per_class,
        random_state=args.random_state,
        n_target_types_per_table=args.n_target_types_per_table,
        column_skip_if_already_present=args.column_skip_if_already_present,
        priority_column_types=args.priority_column_types,
        max_per_column_type=args.max_per_column_type,
        max_per_file_column=args.max_per_file_column,
        limit_files=args.limit_files,
        write_occurrences=args.write_occurrences,
    )


if __name__ == "__main__":
    main()
