"""Build a manifest of disguised unknown / not-applicable cells + matched present cells.

For every column that contains unknown or not-applicable disguised missing
values, includes ALL such missing cells and samples the same number of present
(non-missing) cells from that column (or all available present cells if fewer).

Usage:
    python pipeline/build_token_manifest.py
    python pipeline/build_token_manifest.py --out samples/sample_manifest_unknown_not_applicable.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import REPO_ROOT
from sample import (
    _column_type,
    _detect_co_missing_columns,
    _enriched_is_missing,
    classify_cell,
)

TARGET_GROUPS = frozenset({"unknown", "not applicable"})


def token_group(special_token) -> str | None:
    """Fold special_token labels into thesis groups (unknown / not applicable)."""
    t = "" if pd.isna(special_token) else str(special_token).strip().lower()
    if t.startswith("unknown"):
        return "unknown"
    if t in {"n/a", "na", "not_applicable", "not applicable"}:
        return "not applicable"
    return None


def _col_rng(random_state: int, file_basename: str, target_col: str) -> np.random.Generator:
    file_hash = int.from_bytes(
        hashlib.blake2b(file_basename.encode("utf-8"), digest_size=8).digest(), "big"
    )
    col_hash = int.from_bytes(
        hashlib.blake2b(str(target_col).encode("utf-8"), digest_size=8).digest(), "big"
    )
    return np.random.default_rng((random_state * 1000003) ^ file_hash ^ col_hash)


def build_manifest(
    datasets_dir: Path,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict] = []
    sample_id = 0

    for fp in sorted(datasets_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        file_basename = fp.name
        table_id = fp.stem
        file_rel = f"datasets/{file_basename}"

        for col in df.columns:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            column_type = _column_type(str(col))
            target_col_idx = int(df.columns.get_loc(col))
            miss_mask = _enriched_is_missing(series)

            miss_rows: list[dict] = []
            miss_row_idx: set[int] = set()

            for row_idx in series.index[miss_mask]:
                val = series.at[row_idx]
                missing_kind, special_token, original_value = classify_cell(val, is_numeric)
                if missing_kind != "disguised":
                    continue
                if token_group(special_token) not in TARGET_GROUPS:
                    continue
                row_idx_int = int(row_idx)
                miss_row_idx.add(row_idx_int)
                miss_rows.append({
                    "row_idx": row_idx_int,
                    "special_token": special_token,
                    "original_value": original_value,
                })

            if not miss_rows:
                continue

            dropped_cols = _detect_co_missing_columns(df, col)
            n_miss = len(miss_rows)

            pres_idx = np.array([
                int(r) for r in series.index[~miss_mask] if int(r) not in miss_row_idx
            ])
            n_pres = min(n_miss, len(pres_idx))
            chosen_pres = (
                _col_rng(random_state, file_basename, col).choice(
                    pres_idx, size=n_pres, replace=False
                )
                if n_pres > 0 else np.array([], dtype=int)
            )

            for m in miss_rows:
                rows.append({
                    "sample_id": sample_id,
                    "file": file_rel,
                    "file_basename": file_basename,
                    "table_id": table_id,
                    "column_type": column_type,
                    "target_col": str(col),
                    "target_col_idx": target_col_idx,
                    "row_idx": m["row_idx"],
                    "y": 1,
                    "is_missing": True,
                    "missing_kind": "disguised",
                    "special_token": m["special_token"],
                    "original_value": m["original_value"],
                    "co_missing_columns": dropped_cols,
                    "random_state": random_state,
                })
                sample_id += 1

            for row_idx_int in chosen_pres:
                val = series.at[row_idx_int]
                _, _, original_value = classify_cell(val, is_numeric)
                rows.append({
                    "sample_id": sample_id,
                    "file": file_rel,
                    "file_basename": file_basename,
                    "table_id": table_id,
                    "column_type": column_type,
                    "target_col": str(col),
                    "target_col_idx": target_col_idx,
                    "row_idx": int(row_idx_int),
                    "y": 0,
                    "is_missing": False,
                    "missing_kind": "present",
                    "special_token": None,
                    "original_value": original_value,
                    "co_missing_columns": dropped_cols,
                    "random_state": random_state,
                })
                sample_id += 1

    out = pd.DataFrame(rows)
    if not out.empty:
        out["co_missing_columns"] = out["co_missing_columns"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Flat parquet corpus (default: datasets/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "samples" / "sample_manifest_unknown_not_applicable.parquet",
        help="Output manifest path",
    )
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    manifest = build_manifest(args.datasets, random_state=args.random_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)

    if manifest.empty:
        print(f"[build_token_manifest] no rows matched; wrote empty -> {args.out}")
        return

    grp = manifest.loc[manifest.y == 1, "special_token"].map(token_group).value_counts()
    print(f"[build_token_manifest] wrote {len(manifest):,} rows -> {args.out}")
    print(f"  missing (y=1): {(manifest.y == 1).sum():,}")
    print(f"  present (y=0): {(manifest.y == 0).sum():,}")
    print("  missing by token:")
    print(grp.to_string())
    print(
        f"tables: {manifest.file_basename.nunique():,}  "
        f"columns: {manifest[['file_basename', 'target_col']].drop_duplicates().shape[0]:,}"
    )


if __name__ == "__main__":
    main()
