"""Build a prediction-dataset shard from per-row cell embeddings.

The embedder produces an `(n_rows, hidden_size)` array directly (target
slicing / context-mean / meanpool all happen upstream in step 4). This
module attaches provenance and shapes the row layout to the trimmed
canonical schema (CLEANUP_SPEC §4):

  sample_id, table_id, column_type, row_idx, row_in_table,
  y, special_token, emb_000 ... emb_{H-1}

Dropped vs. the old schema: `file` (== path of `table_id`), `target_col`
(== `column_type` for output purposes) and `missing_kind` (`y` is derived
from it). All three remain recoverable from `sample_manifest.parquet` via
`sample_id`.

Embedding column naming uses `f"emb_{i:0Nd}"` with N = max(3, len(str(H-1)))
-> 3 digits for 384/768, 4 digits for 1024. `y` is derived from
`missing_kind` so labels never disagree with the manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _emb_pad_width(hidden_size: int) -> int:
    return max(3, len(str(hidden_size - 1)))


def build_pred_dataset(
    *,
    cell_embeddings,
    provenance: pd.DataFrame,
    table_id: str,
    column_type: str,
    hidden_size: int,
) -> pd.DataFrame:
    """Assemble one (file, target_col) shard of the prediction dataset.

    Parameters
    ----------
    cell_embeddings : np.ndarray | torch.Tensor
        Shape (n_rows, hidden_size); one row per sampled cell.
    provenance : pd.DataFrame
        Per-row metadata from `02_input_prep`: sample_id, row_in_table,
        row_idx, missing_kind, special_token.
    table_id : str
        Source parquet stem.
    column_type : str
        `_normalize(target_col)` (readable label, not unique).
    hidden_size : int
        Embedding dimensionality (sizes the emb_* columns).
    """
    if hasattr(cell_embeddings, "detach"):  # torch.Tensor
        arr = cell_embeddings.detach().cpu().numpy()
    else:
        arr = np.asarray(cell_embeddings)
    arr = arr.astype("float32")

    n_rows = arr.shape[0]
    if n_rows != len(provenance):
        raise ValueError(
            f"cell_embeddings has {n_rows} rows but provenance has {len(provenance)}"
        )
    if arr.shape[1] != hidden_size:
        raise ValueError(
            f"cell_embeddings hidden dim {arr.shape[1]} != hidden_size {hidden_size}"
        )

    pad = _emb_pad_width(hidden_size)
    emb_cols = [f"emb_{i:0{pad}d}" for i in range(hidden_size)]
    emb_df = pd.DataFrame(arr, columns=emb_cols)

    # y derived from missing_kind ('present' is the ONLY non-missing kind);
    # missing_kind itself is not emitted (recoverable from the manifest).
    y = (provenance["missing_kind"] != "present").astype(int).to_numpy()

    meta_df = pd.DataFrame({
        "sample_id":     provenance["sample_id"].to_numpy(),
        "table_id":      [table_id] * n_rows,
        "column_type":   [column_type] * n_rows,
        "row_idx":       provenance["row_idx"].to_numpy(),
        "row_in_table":  provenance["row_in_table"].to_numpy(),
        "y":             y,
        "special_token": provenance["special_token"].to_numpy(),
    })

    return pd.concat(
        [meta_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1
    )
