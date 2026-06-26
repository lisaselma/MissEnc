"""Random sample-manifest builder (refactor; REFACTOR_SPEC.md step 2).

Walks a FLAT ``datasets/*.parquet`` corpus, applies the enriched
``is_missing(...)`` predicate (NaN OR disguised literal/regex/numeric
sentinel), and writes one canonical ``samples/sample_manifest.parquet``
containing 5 missing + 5 present rows per chosen (file, target_col).

The manifest is the SINGLE source of truth for every downstream
encoder x ablation run. It is the pre-embedding sanity-check artifact:
inspect it before any tokenization happens.

Key behaviours (vs. the old semantic-type sampler):
  * Source discovery is a flat ``datasets/`` directory -- no
    ``by_semantic_type/<type>/`` taxonomy, no canonical/folder priority.
  * Column selection is UNIFORMLY RANDOM over eligible columns (no
    ranking). Default one column per file per run.
  * The manifest is APPEND-ONLY. On every run, an existing manifest is
    loaded automatically: prior (file, row_idx) rows are excluded so a
    physical row is never sampled twice, and ``sample_id`` continues
    from the existing max.
  * Per-file seeds are run-indexed: ``blake2b(basename) XOR run_index``
    where ``run_index`` = how many prior runs already touched the file.
    Reruns therefore explore a different random column (and only ever
    draw from a file's leftover rows).

Also persists, per (file, target_col), the list of context columns that
are perfectly co-missing with the target -- so the ``co_missing_drop``
ablation is a cheap one-line toggle downstream.

Outputs (under --out-dir, default samples/):
  - sample_manifest.parquet            (canonical per-row schema, §2.3)
  - summary_samples.csv                (per-column audit)
  - co_missing_columns_per_table.csv   (per (file, target) co-missing view)
  - sampling_run_metadata.json         (config snapshot + sha256s)
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


def _file_seed(file_basename: str, run_index: int) -> int:
    """Run-indexed per-file seed: ``blake2b(basename) XOR run_index``.

    Same file, next run -> different RNG stream -> likely a different
    random column. Because already-sampled rows are hard-excluded, even
    re-landing on the same column just draws from its leftover rows.
    """
    h = hashlib.blake2b(file_basename.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") ^ int(run_index)


def _enriched_is_missing(series: pd.Series, is_missing_fn) -> pd.Series:
    """Apply is_missing across a column -> boolean Series on the input index."""
    return series.map(is_missing_fn).astype(bool)


# Aggregate-cell pattern: matches strings like
#   "Female:45996; Male:43803; Unknown:157"
#   "0-19:1127; 20-29:3919; 30-39:4716"
# i.e. >=2 "key:value" segments. Such cells encode per-group COUNT
# distributions rather than atomic values; out of scope for cell-level
# missingness.
AGGREGATE_CELL_RE: re.Pattern = re.compile(
    r"^\s*(?:[^:;,]+:\s*<?\s*\d+(?:\.\d+)?\s*%?\s*[;,]?\s*){2,}$"
)


def _column_stats(series: pd.Series, is_missing_fn) -> dict:
    """Per-column diagnostics used by the eligibility filter."""
    miss_mask = _enriched_is_missing(series, is_missing_fn)
    n_m = int(miss_mask.sum())
    n_p = int((~miss_mask).sum())
    n_d = int((miss_mask & ~series.isna()).sum())
    if n_p == 0:
        return {
            "n_missing": n_m, "n_present": 0, "n_disguised": n_d,
            "n_unique_present": 0, "uniq_frac": 0.0,
            "mean_present_len": 0.0, "agg_frac": 0.0,
            "miss_mask": miss_mask,
        }
    present = series[~miss_mask]
    n_u = int(present.nunique(dropna=False))
    uniq_frac = n_u / n_p
    present_str = present.astype("string").fillna("")
    mean_len = float(present_str.str.len().mean())
    n_agg = int(present_str.str.match(AGGREGATE_CELL_RE).sum())
    return {
        "n_missing": n_m, "n_present": n_p, "n_disguised": n_d,
        "n_unique_present": n_u, "uniq_frac": uniq_frac,
        "mean_present_len": mean_len, "agg_frac": n_agg / n_p,
        "miss_mask": miss_mask,
    }


# Parquet/pandas-internal column names that aren't real data.
INTERNAL_COL_RE: re.Pattern = re.compile(
    r"^(?:__index_level_\d+__|__pandas[\w_]*|Unnamed:\s*\d+)$"
)


def _eligibility_reason(
    stats: dict,
    column_name: str,
    min_n_per_class: int,
    aggregate_threshold: float,
    unique_id_max_frac: float,
    free_text_max_mean_len: float,
) -> str | None:
    """None if the column is an eligible target candidate, else a short
    reason string. Screens internal/index, constant, unique-ID,
    free-text, aggregate columns, and any column lacking at least
    ``min_n_per_class`` of BOTH missing and present."""
    if INTERNAL_COL_RE.match(str(column_name)):
        return "internal_column"
    if stats["n_missing"] < min_n_per_class:
        return "lt_n_missing"
    if stats["n_present"] < min_n_per_class:
        return "lt_n_present"
    if stats["n_unique_present"] <= 1:
        return "constant"
    if stats["uniq_frac"] >= unique_id_max_frac:
        return "unique_id"
    if stats["mean_present_len"] > free_text_max_mean_len:
        return "free_text"
    if stats["agg_frac"] >= aggregate_threshold:
        return "aggregate_column"
    return None


def _eligible_columns_for_file(
    df: pd.DataFrame,
    is_missing_fn,
    *,
    min_n_per_class: int,
    aggregate_threshold: float,
    unique_id_max_frac: float,
    free_text_max_mean_len: float,
    file_seed: int,
    skip_columns: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Scan every column and return (eligible, candidates).

    ``eligible`` is the list of eligible-column dicts, SHUFFLED by a
    per-file RNG (no canonical priority, no balance ranking -- §3 step 2,
    invariant 7). ``candidates`` is the per-column audit (one row per
    scanned column, eligible or not).
    """
    skip_columns = skip_columns or set()
    candidates: list[dict] = []
    eligible: list[dict] = []
    for col in df.columns:
        stats = _column_stats(df[col], is_missing_fn)
        reason = _eligibility_reason(
            stats,
            column_name=col,
            min_n_per_class=min_n_per_class,
            aggregate_threshold=aggregate_threshold,
            unique_id_max_frac=unique_id_max_frac,
            free_text_max_mean_len=free_text_max_mean_len,
        )
        if reason is None and str(col) in skip_columns:
            reason = "already_sampled_column"
        row = {
            "column": str(col),
            "n_missing": stats["n_missing"],
            "n_present": stats["n_present"],
            "n_disguised": stats["n_disguised"],
            "uniq_frac": round(stats["uniq_frac"], 4),
            "mean_present_len": round(stats["mean_present_len"], 2),
            "agg_frac": round(stats["agg_frac"], 4),
            "eligibility_reason": reason,
            "miss_mask": stats["miss_mask"],
        }
        candidates.append(row)
        if reason is None:
            eligible.append(row)

    # Uniformly random over eligible columns. Sort first for a stable
    # cross-machine starting order, then shuffle with the run-indexed seed.
    col_rng = np.random.default_rng(file_seed ^ 0xC01_C0DE)
    eligible.sort(key=lambda r: r["column"])
    col_rng.shuffle(eligible)
    return eligible, candidates


def _detect_co_missing_columns(
    df: pd.DataFrame, target_col: str, is_missing_fn
) -> list[str]:
    """Context columns perfectly co-missing with the target under the
    enriched mask. A column ``c`` is co-missing iff
    P(t_missing | c_missing) == 1 AND P(c_missing | t_missing) == 1.
    """
    t_mask = _enriched_is_missing(df[target_col], is_missing_fn)
    n_t = int(t_mask.sum())
    if n_t == 0:
        return []
    dropped: list[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        c_mask = _enriched_is_missing(df[c], is_missing_fn)
        n_c = int(c_mask.sum())
        if n_c == 0:
            continue
        inter = int((t_mask & c_mask).sum())
        if (inter / n_c) == 1.0 and (inter / n_t) == 1.0:
            dropped.append(str(c))
    return dropped


def _load_existing_skip_sets(
    path: Path,
) -> tuple[dict[str, set[int]], dict[str, int], int]:
    """Load the existing manifest (if present) and return:

      (rows_by_file, run_index_by_file, max_sample_id)

    ``rows_by_file[basename]``      -> set of row_idx already sampled.
    ``run_index_by_file[basename]`` -> next run_index to assign (max
                                       prior run_index + 1).
    ``max_sample_id``               -> highest sample_id seen (-1 if none).

    Missing file -> empty skip sets + max_sample_id == -1 (fresh start).
    """
    rows_by_file: dict[str, set[int]] = {}
    run_index_by_file: dict[str, int] = {}
    max_sample_id = -1
    if not path.exists():
        return rows_by_file, run_index_by_file, max_sample_id

    cols = ["sample_id", "file_basename", "row_idx", "run_index"]
    df = pd.read_parquet(path)
    have_run_index = "run_index" in df.columns
    use_cols = [c for c in cols if c in df.columns]
    df = df[use_cols]
    if not df.empty:
        max_sample_id = int(df["sample_id"].max())
    for row in df.itertuples(index=False):
        fb = str(row.file_basename)
        rows_by_file.setdefault(fb, set()).add(int(row.row_idx))
        if have_run_index:
            prev = run_index_by_file.get(fb, -1)
            run_index_by_file[fb] = max(prev, int(row.run_index))
    # Convert "max prior run_index" -> "next run_index".
    run_index_by_file = {fb: v + 1 for fb, v in run_index_by_file.items()}
    return rows_by_file, run_index_by_file, max_sample_id


def _load_occurrences_lookup(occurrences_csv: Path) -> dict:
    """Build (file_basename, column, row_idx) -> disguised info dict."""
    if not occurrences_csv.exists():
        print(f"[00a] WARNING: occurrences CSV not found at {occurrences_csv}; "
              f"disguised provenance will be empty. Run 00b first.")
        return {}
    occ = pd.read_csv(occurrences_csv)
    lookup: dict = {}
    for _, r in occ.iterrows():
        key = (str(r["file"]), str(r["column"]), int(r["row_idx"]))
        lookup[key] = {
            "matched_pattern": str(r["matched_pattern"]) if pd.notna(r["matched_pattern"]) else None,
            "original_value": str(r["original_value"]),
        }
    return lookup


def build_sample_manifest(
    *,
    repo_root: Path,
    datasets_root: Path,
    occurrences_csv: Path,
    out_dir: Path,
    n_per_class: int = 5,
    random_state: int = 42,
    n_target_types_per_table: int = 1,
    limit_files: int | None = None,
    aggregate_threshold: float = 0.10,
    unique_id_max_frac: float = 0.95,
    free_text_max_mean_len: float = 60.0,
    manifest_path: Path | None = None,
) -> Path:
    """Build/extend ``sample_manifest.parquet`` and summaries.

    Returns the manifest path. If a manifest already exists at
    ``manifest_path`` (default: out_dir/sample_manifest.parquet), it is
    loaded automatically as a skip source and the new rows are APPENDED.
    """
    if random_state < 0:
        raise ValueError("random_state must be a non-negative int")
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    if n_target_types_per_table < 1:
        raise ValueError("n_target_types_per_table must be >= 1")

    clean = load_sibling_module("01_table_cleaning.py", alias="_cleaning")
    is_missing_fn = clean.is_missing
    normalize_fn = clean._normalize

    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = out_dir / "sample_manifest.parquet"

    # Auto-append: load any existing manifest as a skip source.
    rows_by_file, run_index_by_file, max_sample_id = _load_existing_skip_sets(manifest_path)
    sample_id_counter = max_sample_id + 1
    if max_sample_id >= 0:
        n_prior_rows = sum(len(v) for v in rows_by_file.values())
        print(f"[00a] append mode: existing manifest has {n_prior_rows:,} rows "
              f"across {len(rows_by_file)} files; continuing sample_id at {sample_id_counter}")

    print(
        f"[00a] config: n_per_class={n_per_class}, "
        f"n_target_types_per_table={n_target_types_per_table}, "
        f"random_state={random_state}"
    )

    occurrences_lookup = _load_occurrences_lookup(occurrences_csv)

    parquet_files = sorted(datasets_root.glob("*.parquet"))
    if limit_files is not None:
        parquet_files = parquet_files[:limit_files]
    print(f"[00a] {len(parquet_files)} parquet files under {datasets_root}")

    manifest_rows: list[dict] = []
    summary_rows: list[dict] = []
    co_missing_rows: list[dict] = []
    n_sampled_groups = 0
    n_skipped_groups = 0

    for i, fp in enumerate(parquet_files, 1):
        file_basename = fp.name
        file_rel = str(fp.relative_to(repo_root))
        table_id = fp.stem
        run_index = run_index_by_file.get(file_basename, 0)

        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  [{i}/{len(parquet_files)}] SKIP {file_basename}: read error: {e}")
            summary_rows.append({
                "file": file_basename, "target_col": "",
                "column_type": "", "n_rows_total": 0,
                "n_missing_total": 0, "n_present_total": 0, "n_disguised_total": 0,
                "uniq_frac": None, "mean_present_len": None, "aggregate_frac": None,
                "sampled": False, "n_sampled_missing": 0, "n_sampled_present": 0,
                "skip_reason": "read_error", "co_missing_n": 0, "run_index": run_index,
            })
            continue

        file_seed = _file_seed(file_basename, run_index)
        # Rows already used for this file (prior runs); within this run we
        # also avoid reusing a row across multiple picked columns.
        used_row_idx_in_file: set[int] = set(rows_by_file.get(file_basename, set()))

        eligible, candidates = _eligible_columns_for_file(
            df,
            is_missing_fn,
            min_n_per_class=n_per_class,
            aggregate_threshold=aggregate_threshold,
            unique_id_max_frac=unique_id_max_frac,
            free_text_max_mean_len=free_text_max_mean_len,
            file_seed=file_seed,
            skip_columns=None,
        )

        # Audit rows for ineligible columns up front.
        for cand in candidates:
            if cand["eligibility_reason"] is None:
                continue
            summary_rows.append({
                "file": file_basename, "target_col": cand["column"],
                "column_type": normalize_fn(cand["column"]),
                "n_rows_total": len(df),
                "n_missing_total": cand["n_missing"], "n_present_total": cand["n_present"],
                "n_disguised_total": cand["n_disguised"], "uniq_frac": cand["uniq_frac"],
                "mean_present_len": cand["mean_present_len"], "aggregate_frac": cand["agg_frac"],
                "sampled": False, "n_sampled_missing": 0, "n_sampled_present": 0,
                "skip_reason": cand["eligibility_reason"], "co_missing_n": 0,
                "run_index": run_index,
            })

        if not eligible:
            n_skipped_groups += 1
            if i % 50 == 0 or i == len(parquet_files):
                print(f"  [{i}/{len(parquet_files)}] sampled groups: {n_sampled_groups}, "
                      f"manifest rows: {len(manifest_rows)}")
            continue

        # Try eligible columns in random order until we get
        # n_target_types_per_table SUCCESSFUL picks (or exhaust them).
        n_picked = 0
        for picked_entry in eligible:
            if n_picked >= n_target_types_per_table:
                break
            target_col = picked_entry["column"]
            is_miss_mask = picked_entry["miss_mask"]
            dropped_cols = _detect_co_missing_columns(df, target_col, is_missing_fn)

            # Per-target RNG: mix file seed with a stable hash of the column.
            col_hash = int.from_bytes(
                hashlib.blake2b(str(target_col).encode("utf-8"), digest_size=8).digest(), "big",
            )
            rng = np.random.default_rng((file_seed * 1000003) ^ col_hash)

            miss_idx = np.array([
                int(ix) for ix in df.index[is_miss_mask] if int(ix) not in used_row_idx_in_file
            ])
            pres_idx = np.array([
                int(ix) for ix in df.index[~is_miss_mask] if int(ix) not in used_row_idx_in_file
            ])

            if len(miss_idx) < n_per_class or len(pres_idx) < n_per_class:
                summary_rows.append({
                    "file": file_basename, "target_col": target_col,
                    "column_type": normalize_fn(target_col), "n_rows_total": len(df),
                    "n_missing_total": picked_entry["n_missing"],
                    "n_present_total": picked_entry["n_present"],
                    "n_disguised_total": picked_entry["n_disguised"],
                    "uniq_frac": picked_entry["uniq_frac"],
                    "mean_present_len": picked_entry["mean_present_len"],
                    "aggregate_frac": picked_entry["agg_frac"],
                    "sampled": False, "n_sampled_missing": 0, "n_sampled_present": 0,
                    "skip_reason": "insufficient_rows_after_exclusion",
                    "co_missing_n": len(dropped_cols), "run_index": run_index,
                })
                n_skipped_groups += 1
                continue

            chosen_miss = rng.choice(miss_idx, size=n_per_class, replace=False)
            chosen_pres = rng.choice(pres_idx, size=n_per_class, replace=False)
            used_row_idx_in_file.update(int(x) for x in chosen_miss)
            used_row_idx_in_file.update(int(x) for x in chosen_pres)

            chosen = np.empty(2 * n_per_class, dtype=chosen_miss.dtype)
            chosen[0::2] = chosen_miss
            chosen[1::2] = chosen_pres
            target_col_idx = int(df.columns.get_loc(target_col))
            column_type = normalize_fn(target_col)

            for row_idx in chosen:
                row_idx_int = int(row_idx)
                raw_val = df.at[row_idx_int, target_col]
                is_nan = bool(pd.isna(raw_val))
                is_miss = bool(is_miss_mask.loc[row_idx_int])

                if not is_miss:
                    missing_kind = "present"
                    special_token = None
                    original_value = str(raw_val)
                elif is_nan:
                    missing_kind = "nan"
                    special_token = None
                    original_value = None
                else:
                    missing_kind = "disguised"
                    occ = occurrences_lookup.get((file_basename, str(target_col), row_idx_int))
                    original_value = occ["original_value"] if occ else str(raw_val)
                    special_token = original_value

                manifest_rows.append({
                    "sample_id": sample_id_counter,
                    "file": file_rel,
                    "file_basename": file_basename,
                    "table_id": table_id,
                    "column_type": column_type,
                    "target_col": str(target_col),
                    "target_col_idx": target_col_idx,
                    "row_idx": row_idx_int,
                    "y": int(is_miss),
                    "is_missing": is_miss,
                    "missing_kind": missing_kind,
                    "special_token": special_token,
                    "original_value": original_value,
                    "co_missing_columns": dropped_cols,
                    "run_index": run_index,
                    "random_state": random_state,
                })
                sample_id_counter += 1

            summary_rows.append({
                "file": file_basename, "target_col": target_col,
                "column_type": column_type, "n_rows_total": len(df),
                "n_missing_total": picked_entry["n_missing"],
                "n_present_total": picked_entry["n_present"],
                "n_disguised_total": picked_entry["n_disguised"],
                "uniq_frac": picked_entry["uniq_frac"],
                "mean_present_len": picked_entry["mean_present_len"],
                "aggregate_frac": picked_entry["agg_frac"],
                "sampled": True, "n_sampled_missing": n_per_class,
                "n_sampled_present": n_per_class, "skip_reason": None,
                "co_missing_n": len(dropped_cols), "run_index": run_index,
            })
            co_missing_rows.append({
                "file": file_basename, "target_col": target_col,
                "column_type": column_type,
                "co_missing_columns": json.dumps(dropped_cols, ensure_ascii=False),
                "n_co_missing": len(dropped_cols), "total_cols": int(df.shape[1]),
            })
            n_sampled_groups += 1
            n_picked += 1

        if i % 50 == 0 or i == len(parquet_files):
            print(f"  [{i}/{len(parquet_files)}] sampled groups: {n_sampled_groups}, "
                  f"manifest rows: {len(manifest_rows)}")

    # ------------------------------------------------------------------
    # Merge with the existing manifest (append-only) and persist.
    # ------------------------------------------------------------------
    new_df = pd.DataFrame(manifest_rows)
    if not new_df.empty:
        new_df["co_missing_columns"] = new_df["co_missing_columns"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

    if manifest_path.exists():
        existing_df = pd.read_parquet(manifest_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True) if not new_df.empty else existing_df
    else:
        combined = new_df

    summary_df = pd.DataFrame(summary_rows)
    co_missing_df = pd.DataFrame(co_missing_rows)

    tmp_path = manifest_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(manifest_path)

    # Summaries: append CSVs so the audit trail accumulates across runs.
    summary_path = out_dir / "summary_samples.csv"
    co_missing_path = out_dir / "co_missing_columns_per_table.csv"
    summary_df.to_csv(
        summary_path, mode="a", index=False, header=not summary_path.exists()
    )
    if not co_missing_df.empty:
        co_missing_df.to_csv(
            co_missing_path, mode="a", index=False, header=not co_missing_path.exists()
        )

    # Invariant check: exactly n_per_class missing == present per NEW group.
    if not new_df.empty:
        for (f, tc), g in new_df.groupby(["file", "target_col"], sort=False):
            n_miss = int(g["is_missing"].sum())
            n_pres = int((~g["is_missing"]).sum())
            if not (n_miss == n_pres == n_per_class):
                print(f"[00a] INVARIANT VIOLATION on ({f}, {tc}): "
                      f"miss={n_miss}, pres={n_pres}, expected {n_per_class}/{n_per_class}")

    metadata = {
        "scanned_root": str(datasets_root.relative_to(repo_root)),
        "n_files_scanned": len(parquet_files),
        "n_groups_sampled_this_run": n_sampled_groups,
        "n_groups_skipped_this_run": n_skipped_groups,
        "n_rows_added_this_run": int(len(new_df)),
        "n_rows_in_manifest_total": int(len(combined)),
        "n_per_class": n_per_class,
        "n_target_types_per_table": n_target_types_per_table,
        "aggregate_threshold": aggregate_threshold,
        "unique_id_max_frac": unique_id_max_frac,
        "free_text_max_mean_len": free_text_max_mean_len,
        "random_state": random_state,
        "occurrences_csv": str(occurrences_csv),
        "disguised_lookup_size": len(occurrences_lookup),
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": sha256(manifest_path),
        "git_sha": git_sha(REPO_ROOT),
        "finished_at": now_iso(),
        "host": __import__("platform").node(),
        "python": sys.version.split()[0],
    }
    meta_path = out_dir / "sampling_run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"[00a] wrote manifest -> {manifest_path}  "
          f"(+{len(new_df):,} rows this run, {len(combined):,} total, "
          f"{n_sampled_groups} groups sampled, {n_skipped_groups} skipped)")
    return manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "datasets",
                   help="Flat directory of *.parquet (default: repo/datasets)")
    p.add_argument("--occurrences-csv", type=Path,
                   default=REPO_ROOT / "samples" / "special_tokens" / "occurrences_special_tokens.csv",
                   help="00b occurrences CSV (default: samples/special_tokens/occurrences_special_tokens.csv)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "samples",
                   help="Output directory (default: samples)")
    p.add_argument("--n-per-class", type=int, default=5,
                   help="Missing AND present rows per chosen column (default: 5). "
                        "A column with fewer than this in either class (after "
                        "excluding already-sampled rows) is skipped.")
    p.add_argument("--n-target-types-per-table", type=int, default=1,
                   help="Number of random eligible columns to sample per file "
                        "per run (default: 1). Raise to sample more per run.")
    p.add_argument("--random-state", type=int, default=42,
                   help="Base seed, XOR'd with each file's run_index (default: 42)")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Smoke test: only scan the first N parquet files")
    p.add_argument("--aggregate-threshold", type=float, default=0.10)
    p.add_argument("--unique-id-max-frac", type=float, default=0.95)
    p.add_argument("--free-text-max-mean-len", type=float, default=60.0)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    build_sample_manifest(
        repo_root=REPO_ROOT,
        datasets_root=args.datasets_root,
        occurrences_csv=args.occurrences_csv,
        out_dir=args.out_dir,
        n_per_class=args.n_per_class,
        random_state=args.random_state,
        n_target_types_per_table=args.n_target_types_per_table,
        limit_files=args.limit_files,
        aggregate_threshold=args.aggregate_threshold,
        unique_id_max_frac=args.unique_id_max_frac,
        free_text_max_mean_len=args.free_text_max_mean_len,
    )


if __name__ == "__main__":
    main()
