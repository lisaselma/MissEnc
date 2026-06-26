"""Unified encoder-input preparation (refactor; spec s3 step 3).

One parameterized transform turns a (file, target_col) slice of
`sample_manifest.parquet` plus its source table into the tokenizer input
its encoder needs. The *ablation* decides header / target / co-missing
handling; the *encoder family* decides the output FORM:

  TAPAS  -> a sliced+masked table (DataFrame) ready for TapasTokenizer.
  MiniLM -> one serialized sentence string per row.

Sampling, labelling and masking decisions were made in 00a; this module
is a pure deterministic transform. `y` is never recomputed here (it comes
from the manifest's `missing_kind`).

Ablation semantics (applied before tokenization for both families):
  default          : headers preserved; target -> [?]
  without_header   : TAPAS rename cols c0,c1,..; MiniLM omit column names
  without_target   : drop target entirely (no [?]); TAPAS row emb = mean of
                     remaining cells (handled in step 4); MiniLM sentence
                     simply omits the target segment
  only_target      : all cols -> c0,c1,.. EXCEPT target keeps its cleaned name
  meanpool_target  : build the default context input + expose the target
                     column name; the 50/50 pool happens in step 4
  co_missing_drop  : drop the stored co_missing_columns, then == default
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from _pipeline_utils import load_sibling_module


# ---------------------------------------------------------------------------
# Masking / token routing lives HERE (and only here): by the locked step
# order, tokens are introduced after sampling, before tokenization. The
# cleaning module (01_table_cleaning.py) owns detection only; this module
# owns substitution.
#
#   MASK_TOKEN     routes the TARGET cell.
#   EMPTY_TOKEN    routes NON-target missing/disguised cells.
#   EMPTY_SENTINEL is the "n/a" string TAPAS routes to [EMPTY] natively
#                  (TapasTokenizer drops truly-empty strings; "n/a"
#                  survives and collapses to the [EMPTY] special token).
# MiniLM has no native empty handling, so for MiniLM serialization the
# literal EMPTY_TOKEN / MASK_TOKEN strings are emitted and added as atomic
# tokens (embedding matrix resized in 04_build_embeddings.MiniLMCellEmbedder).
# ---------------------------------------------------------------------------
MASK_TOKEN: str = "[?]"
EMPTY_TOKEN: str = "[EMPTY]"
EMPTY_SENTINEL: str = "n/a"


def canonicalize_missing(
    df: pd.DataFrame,
    *,
    skip_columns: set[str] | None = None,
    sentinel: str = EMPTY_SENTINEL,
    is_missing_fn: Callable | None = None,
) -> pd.DataFrame:
    """Replace every is_missing(cell) value with `sentinel` (default 'n/a').

    Operates on a deep copy; input frame is never mutated. Cells in
    `skip_columns` are left untouched (typically: the target column, which
    is overwritten with '[?]' separately). Returns an object-dtype
    DataFrame so all cells are strings ready for TAPAS tokenization. The
    missingness predicate defaults to `01_table_cleaning.is_missing`.
    """
    if is_missing_fn is None:
        is_missing_fn = _load_cleaning_helpers().is_missing
    df = df.copy()
    skip = set(skip_columns or ())

    for col in df.columns:
        if col in skip:
            continue
        df[col] = df[col].apply(
            lambda v, _s=sentinel: _s if is_missing_fn(v) else v
        )

    return df.astype(str)


@dataclass(frozen=True)
class AblationSpec:
    """Minimal ablation contract consumed by this module."""
    id: str
    header: str          # "preserve" | "strip" | "only_target"
    target: str          # "mask" | "remove" | "meanpool"
    co_missing_drop: bool


@dataclass
class PreparedGroup:
    """Output of input-prep for one (file, target_col) group."""
    family: str                       # "tapas" | "minilm"
    provenance: pd.DataFrame
    target_col_name: str              # cleaned target name (meanpool partner)
    # TAPAS branch:
    masked: pd.DataFrame | None = None
    target_col_idx: int | None = None  # None when target removed
    # MiniLM branch:
    sentences: list[str] | None = None


def _load_cleaning_helpers():
    return load_sibling_module("01_table_cleaning.py", alias="_cleaning_fallback")


def parse_co_missing_columns(raw) -> list[str]:
    """The manifest stores co_missing_columns as a JSON string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    return json.loads(raw)


def _build_provenance(manifest_slice: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id":      manifest_slice["sample_id"].to_numpy(),
        "row_in_table":   range(n_rows),
        "row_idx":        manifest_slice["row_idx"].to_numpy(),
        "missing_kind":   manifest_slice["missing_kind"].to_numpy(),
        "special_token":  manifest_slice["special_token"].to_numpy(),
    })


def _apply_co_missing_drop(
    df: pd.DataFrame, manifest_slice: pd.DataFrame, ablation: AblationSpec
) -> pd.DataFrame:
    if not ablation.co_missing_drop:
        return df
    dropped = parse_co_missing_columns(manifest_slice["co_missing_columns"].iloc[0])
    present = [c for c in dropped if c in df.columns]
    return df.drop(columns=present) if present else df


# ---------------------------------------------------------------------------
# TAPAS branch
# ---------------------------------------------------------------------------
def prepare_tapas_inputs(
    *,
    df: pd.DataFrame,
    manifest_slice: pd.DataFrame,
    ablation: AblationSpec,
    normalize_fn: Callable | None = None,
    normalize_columns_fn: Callable | None = None,
    canonicalize_missing_fn: Callable | None = None,
    output_dir: Path | None = None,
    save_raw: bool = False,
) -> PreparedGroup:
    if manifest_slice.empty:
        raise ValueError("manifest_slice is empty; nothing to prepare")

    if normalize_fn is None or normalize_columns_fn is None:
        helpers = _load_cleaning_helpers()
        normalize_fn = normalize_fn or helpers._normalize
        normalize_columns_fn = normalize_columns_fn or helpers._normalize_columns
    # Masking belongs to this module; default to the local routine.
    canonicalize_missing_fn = canonicalize_missing_fn or canonicalize_missing

    # `target_col` in the manifest is the CLEANED label and may differ from
    # the on-disk header for collision-renamed columns (e.g. manifest "age"
    # vs dataset "age 2"). The authoritative pointer is `target_col_idx`, the
    # physical column position. Resolve by index, fall back to name lookup
    # for older manifests without the index.
    target_label = manifest_slice["target_col"].iloc[0]
    df = _apply_co_missing_drop(df, manifest_slice, ablation)

    target_pos = None
    if "target_col_idx" in manifest_slice.columns:
        ti = manifest_slice["target_col_idx"].iloc[0]
        if pd.notna(ti) and 0 <= int(ti) < df.shape[1]:
            target_pos = int(ti)
    if target_pos is None:
        if target_label not in df.columns:
            raise KeyError(
                f"target_col {target_label!r} (idx unavailable) not in source "
                f"columns: {list(df.columns)}"
            )
        target_pos = int(list(df.columns).index(target_label))

    # The actual on-disk name at that position drives all df operations;
    # the manifest label is only used for output metadata.
    target_col = df.columns[target_pos]

    row_idxs = manifest_slice["row_idx"].tolist()
    sampled = df.loc[row_idxs].reset_index(drop=True)
    drop_target = ablation.target == "remove"

    masked = sampled.copy()
    masked = canonicalize_missing_fn(masked, skip_columns={target_col})
    if not drop_target:
        masked[target_col] = "[?]"
    for col in masked.columns:
        if col == target_col:
            continue
        masked[col] = masked[col].apply(normalize_fn)
    masked = normalize_columns_fn(masked)  # column names -> _normalize(name)

    target_norm = normalize_fn(target_col)

    if drop_target:
        # Drop by position so duplicate normalized names don't bite.
        masked = masked.drop(masked.columns[target_pos], axis=1)
        target_col_idx = None
    else:
        target_col_idx = target_pos

    # Header rewriting AFTER masking/normalization.
    if ablation.header == "strip":
        masked.columns = [f"c{i}" for i in range(len(masked.columns))]
    elif ablation.header == "only_target":
        new_cols = []
        for i in range(len(masked.columns)):
            if (not drop_target) and i == target_col_idx:
                new_cols.append(target_norm)
            else:
                new_cols.append(f"c{i}")
        masked.columns = new_cols
    # preserve: leave normalized names as-is

    # TAPAS writes Cell(...) objects via iloc; force object dtype.
    masked = masked.astype(object)

    provenance = _build_provenance(manifest_slice, len(sampled))

    if save_raw and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(target_col))
        sampled.to_csv(output_dir / f"{safe}_sampled.csv", index=False)
        masked.to_csv(output_dir / f"{safe}_masked.csv", index=False)

    return PreparedGroup(
        family="tapas",
        provenance=provenance,
        target_col_name=target_norm,
        masked=masked,
        target_col_idx=target_col_idx,
    )


# ---------------------------------------------------------------------------
# MiniLM branch
# ---------------------------------------------------------------------------
def prepare_minilm_inputs(
    *,
    df: pd.DataFrame,
    manifest_slice: pd.DataFrame,
    ablation: AblationSpec,
    normalize_fn: Callable | None = None,
    is_missing_fn: Callable | None = None,
    mask_token: str | None = None,
    empty_token: str | None = None,
    format_cell_fn: Callable | None = None,
) -> PreparedGroup:
    if manifest_slice.empty:
        raise ValueError("manifest_slice is empty; nothing to prepare")

    helpers = None
    if normalize_fn is None or is_missing_fn is None or format_cell_fn is None:
        helpers = _load_cleaning_helpers()
        normalize_fn = normalize_fn or helpers._normalize
        is_missing_fn = is_missing_fn or helpers.is_missing
        format_cell_fn = format_cell_fn or helpers.format_cell_value
    # Tokens are defined in this module (masking happens here, not in cleaning).
    mask_token = mask_token or MASK_TOKEN
    empty_token = empty_token or EMPTY_TOKEN

    target_col = manifest_slice["target_col"].iloc[0]
    df = _apply_co_missing_drop(df, manifest_slice, ablation)
    if target_col not in df.columns:
        raise KeyError(
            f"target_col {target_col!r} not in source columns: {list(df.columns)}"
        )

    row_idxs = manifest_slice["row_idx"].tolist()
    sampled = df.loc[row_idxs].reset_index(drop=True)
    cols = list(sampled.columns)
    target_norm = normalize_fn(target_col)
    drop_target = ablation.target == "remove"

    sentences: list[str] = []
    for _, row in sampled.iterrows():
        parts: list[str] = []
        for i, col in enumerate(cols):
            if col == target_col:
                if drop_target:
                    continue
                # Target masked with [?].
                if ablation.header == "strip":
                    parts.append(mask_token)
                elif ablation.header == "only_target":
                    parts.append(f"{target_norm}: {mask_token}")
                else:  # preserve
                    parts.append(f"{target_norm}: {mask_token}")
            else:
                cell = row[col]
                val = empty_token if is_missing_fn(cell) else format_cell_fn(cell)
                if ablation.header == "strip":
                    parts.append(val)
                elif ablation.header == "only_target":
                    parts.append(f"c{i}: {val}")
                else:  # preserve
                    parts.append(f"{normalize_fn(col)}: {val}")
        sentences.append(", ".join(parts))

    provenance = _build_provenance(manifest_slice, len(sampled))
    return PreparedGroup(
        family="minilm",
        provenance=provenance,
        target_col_name=target_norm,
        sentences=sentences,
    )


def prepare_inputs(
    *,
    family: str,
    df: pd.DataFrame,
    manifest_slice: pd.DataFrame,
    ablation: AblationSpec,
    **kwargs,
) -> PreparedGroup:
    """Dispatch to the TAPAS or MiniLM branch by encoder family."""
    if family == "tapas":
        return prepare_tapas_inputs(
            df=df, manifest_slice=manifest_slice, ablation=ablation, **kwargs
        )
    if family == "minilm":
        return prepare_minilm_inputs(
            df=df, manifest_slice=manifest_slice, ablation=ablation, **kwargs
        )
    raise ValueError(f"unknown encoder family: {family!r}")
