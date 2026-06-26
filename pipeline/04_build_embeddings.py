"""Build per-row cell embeddings.

Ports the TAPAS model wrapper (cell 10) and `get_tapas_cell_embeddings`
(cell 14) from cell_level/prepare_table.ipynb. The class configuration is
loaded from `pipeline/config.yaml` so the central TapasModel definition
stays in one place.

The pipeline encodes ONE row per ablation (target-column slice); there is
no whole-table encoding. `TapasCellEmbedder` therefore encodes row by row
(chunk_size = 1) and `MiniLMCellEmbedder` serializes one sentence per row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
import yaml


_CONFIG_CACHE: dict | None = None


def _first_tapas_encoder(cfg: dict) -> dict:
    """Return the first tapas-family encoder dict from config, or a
    sensible default if the config predates the encoders[] block."""
    for enc in cfg.get("encoders", []):
        if enc.get("family") == "tapas":
            return enc
    return {"name": "google/tapas-base", "max_length": 512, "select_one_column": False}


def _load_config(config_path: str | Path | None = None) -> dict:
    """Load and cache the pipeline config.yaml from disk."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    path = Path(config_path) if config_path else Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if config_path is None:
        _CONFIG_CACHE = cfg
    return cfg




class TapasModel:
    def __init__(self, model_name: str | None = None, select_one_column: bool | None = None):
        """Load a base TAPAS encoder (no QA head).

        Choice of `google/tapas-base` or `google/tapas-large` (the QA head
        is intentionally dropped per the thesis, so no WTQ-finetuned
        variant is used). Defaults are read from the first tapas-family
        entry in pipeline/config.yaml's `encoders:` block.
        """
        cfg = _first_tapas_encoder(_load_config())
        if model_name is None:
            model_name = cfg["name"]
        if select_one_column is None:
            select_one_column = cfg.get("select_one_column", False)

        print("loading tapas model")
        config = transformers.TapasConfig.from_pretrained(model_name)
        config.select_one_column = select_one_column
        self.tokenizer = transformers.TapasTokenizer.from_pretrained(model_name)
        self.model = transformers.TapasModel.from_pretrained(model_name, config=config)
        self.model_name = model_name

        # Register "[?]" as a distinct special token (separate from "[EMPTY]")
        # so masked target cells are not collapsed to the EMPTY token by
        # tokenization_tapas.format_text. New token gets a fresh (untrained) row.
        import transformers.models.tapas.tokenization_tapas as _tt
        if not getattr(_tt.format_text, "_keeps_qmark", False):
            _orig_format_text = _tt.format_text
            def _format_text_keep_qmark(text):
                if isinstance(text, str) and text.strip() == "[?]":
                    return "[?]"
                return _orig_format_text(text)
            _format_text_keep_qmark._keeps_qmark = True
            _tt.format_text = _format_text_keep_qmark

        # pandas >=2 no longer falls back to positional access when a
        # Series is indexed by an int that isn't in its label index, so
        # TAPAS' `row[col_index]` (row from iterrows(), col_index an int)
        # raises KeyError. Patch both call sites to use positional .iloc.
        if not getattr(_tt._get_column_values, "_pos_patched", False):
            def _free_get_column_values(table, col_index):
                index_to_values: dict = {}
                for row_index, row in table.iterrows():
                    text = _tt.normalize_for_match(row.iloc[col_index].text)
                    index_to_values[row_index] = list(_tt._get_numeric_values(text))
                return index_to_values
            _free_get_column_values._pos_patched = True
            _tt._get_column_values = _free_get_column_values
        _orig_method = _tt.TapasTokenizer._get_column_values
        if not getattr(_orig_method, "_pos_patched", False):
            def _method_get_column_values(self, table, col_index):
                table_numeric_values: dict = {}
                for row_index, row in table.iterrows():
                    cell = row.iloc[col_index]
                    if cell.numeric_value is not None:
                        table_numeric_values[row_index] = cell.numeric_value
                return table_numeric_values
            _method_get_column_values._pos_patched = True
            _tt.TapasTokenizer._get_column_values = _method_get_column_values
        existing_additional = getattr(self.tokenizer, "additional_special_tokens", None)
        if existing_additional is None:
            existing_additional = self.tokenizer.special_tokens_map.get(
                "additional_special_tokens", []
            )
        if "[?]" not in existing_additional:
            import inspect
            add_kwargs: dict = {}
            params = inspect.signature(self.tokenizer.add_special_tokens).parameters
            if "replace_additional_special_tokens" in params:
                add_kwargs["replace_additional_special_tokens"] = False
            elif "replace_extra_special_tokens" in params:
                add_kwargs["replace_extra_special_tokens"] = False
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["[?]"]},
                **add_kwargs,
            )
            self.model.resize_token_embeddings(len(self.tokenizer))


def get_tapas_cell_embeddings(
    inputs: dict[str, torch.Tensor], last_hidden_states: torch.Tensor
) -> torch.Tensor:
    """Get the cell embeddings from the last hidden states of a TAPAS model.

    Args:
        inputs: The inputs to the model.
        last_hidden_states: The last hidden states of the model.

    Returns:
        cell_embeddings: The cell embeddings.
    """
    column_ids = inputs["token_type_ids"][0][:, 1]
    row_ids = inputs["token_type_ids"][0][:, 2]

    max_column_id = column_ids.max().item()
    max_row_id = row_ids.max().item()

    cell_embeddings = torch.zeros(
        max_row_id + 1,
        max_column_id,
        last_hidden_states.shape[-1],
        device=last_hidden_states.device,
    )

    for row_id in range(max_row_id + 1):
        for column_id in range(1, max_column_id + 1):
            indices = torch.where(
                (column_ids == column_id) & (row_ids == row_id)
            )[0]

            if len(indices) == 0:
                continue

            embeddings = last_hidden_states[0][indices]

            cell_embeddings[row_id, column_id - 1] = embeddings.mean(dim=0)

    return cell_embeddings


# ---------------------------------------------------------------------------
# v2 additions (§5.6, §5.6.1): BaseEmbedder + TapasCellEmbedder adapter +
# optional HDF5/fp16 attention writer.
#
# Everything above this line is LEGACY (locked, do not modify). The
# adapter below calls into the legacy functions and reduces output to
# (n_rows, hidden_size). The HDF5 writer is opt-in via save_attentions.
# ---------------------------------------------------------------------------


class BaseEmbedder:
    """Abstract embedder for the cell-level pipeline.

    Future non-cell-level models implement this same contract; main.py
    talks only to `BaseEmbedder.encode_table`.
    """
    hidden_size: int
    model_name: str

    def encode_table(
        self,
        *,
        masked: pd.DataFrame,
        target_col_idx: int,
        sample_ids: list[int] | None = None,
        save_attentions: bool = False,
        attentions_h5_path: Path | None = None,
        provenance_for_attn: pd.DataFrame | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Returns (cell_embeddings_for_target_col, extra).

        cell_embeddings_for_target_col : torch.Tensor
            shape (n_rows, hidden_size), one row per masked-table row in
            input order.
        extra : dict
            optional artefacts (e.g. {'chunks_inputs': ..., 'chunks_outputs': ...})
            for debugging; do NOT rely on its content in production.
        """
        raise NotImplementedError


class TapasCellEmbedder(BaseEmbedder):
    """Adapter wrapping the legacy `TapasModel` + chunked encoding loop.

    Reduces each chunk's (rows+1, n_cols, hidden) tensor to the target
    column only (slicing `emb[1:, target_col_idx]`), then concatenates
    across chunks. When `save_attentions=True`, persists per-sample
    attentions to an HDF5 file using fp16 + row-renorm + chunked gzip
    (locked layout from §5.6.1).
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        max_length: int | None = None,
        select_one_column: bool | None = None,
        queries: str = "",
        attention_layers_to_keep: list[int] | None = None,
        attention_heads_to_keep: list[int] | None = None,
        normalize: str = "row_softmax_preserved",
        gzip_level: int = 4,
    ):
        cfg = _load_config()
        # Resolve model settings from the encoders[] block (first
        # tapas-family entry) when the caller doesn't pass them.
        if model_name is None:
            ecfg = _first_tapas_encoder(cfg)
            model_name = ecfg["name"]
            if max_length is None:
                max_length = ecfg.get("max_length", 512)
            if select_one_column is None:
                select_one_column = ecfg.get("select_one_column", False)

        if max_length is None:
            max_length = 512

        self._tapas = TapasModel(model_name=model_name, select_one_column=select_one_column)
        self.tokenizer = self._tapas.tokenizer
        self.model = self._tapas.model
        self.model_name = self._tapas.model_name
        self.max_length = int(max_length)
        self._queries = queries or ""

        # Resolve hidden_size from the actual model so the value is
        # never out of sync with the loaded weights.
        self.hidden_size = int(self.model.config.hidden_size)
        self.n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        self.n_heads = int(getattr(self.model.config, "num_attention_heads", 0))

        self.attention_layers_to_keep = attention_layers_to_keep
        self.attention_heads_to_keep = attention_heads_to_keep
        self.normalize_mode = normalize
        self.gzip_level = int(gzip_level)

    # -- helpers --------------------------------------------------------
    def _resolve_layer_idx(self, n_layers: int) -> list[int]:
        if self.attention_layers_to_keep is None:
            return list(range(n_layers))
        return [(i if i >= 0 else n_layers + i) for i in self.attention_layers_to_keep]

    def _resolve_head_idx(self, n_heads: int) -> list[int]:
        if self.attention_heads_to_keep is None:
            return list(range(n_heads))
        return [(i if i >= 0 else n_heads + i) for i in self.attention_heads_to_keep]

    # -- public API ----------------------------------------------------
    def encode_table(
        self,
        *,
        masked: pd.DataFrame,
        target_col_idx: int | None,
        sample_ids: list[int] | None = None,
        save_attentions: bool = False,
        attentions_h5_path: Path | None = None,
        provenance_for_attn: pd.DataFrame | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Encode each masked row to a single (hidden_size,) vector.

        When ``target_col_idx`` is an int, the row embedding is the target
        cell's embedding (slicing the target column). When it is ``None``
        (without_target ablation), the row embedding is the mean over ALL
        remaining data cells, reproducing the legacy `_meanpool` path.
        """
        queries = self._queries

        chunks_inputs: list[Any] = []
        chunks_outputs: list[Any] = []
        chunk_size = 1  # locked: per-row encoding (matches legacy)

        for start in range(0, len(masked), chunk_size):
            chunk = masked.iloc[start : start + chunk_size].reset_index(drop=True)
            inputs = self.tokenizer(
                table=chunk,
                queries=queries,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=save_attentions)
            chunks_inputs.append(inputs)
            chunks_outputs.append(outputs)

        # Legacy per-chunk reduction to (rows+1, n_cols, hidden).
        chunk_embs = [
            get_tapas_cell_embeddings(inp, out.last_hidden_state)
            for inp, out in zip(chunks_inputs, chunks_outputs)
        ]
        if target_col_idx is None:
            # without_target: mean over all data cells (drop header row).
            per_row = []
            for emb in chunk_embs:
                if emb.shape[0] < 2:
                    raise RuntimeError(
                        f"expected header+data rows, got shape {tuple(emb.shape)}"
                    )
                per_row.append(emb[1:, :, :].mean(dim=(0, 1)))
            cell_embeddings = torch.stack(per_row, dim=0).detach().cpu()
        else:
            per_row_target = [emb[1:, target_col_idx, :] for emb in chunk_embs]
            cell_embeddings = torch.cat(per_row_target, dim=0).detach().cpu()
        # sanity: one row per masked row
        if cell_embeddings.shape[0] != len(masked):
            raise RuntimeError(
                f"Embedding row count {cell_embeddings.shape[0]} != "
                f"len(masked) {len(masked)} for target_col_idx={target_col_idx}"
            )

        if save_attentions:
            if attentions_h5_path is None:
                raise ValueError("attentions_h5_path is required when save_attentions=True")
            self._persist_attentions(
                attentions_h5_path=Path(attentions_h5_path),
                chunks_outputs=chunks_outputs,
                sample_ids=sample_ids,
                provenance_for_attn=provenance_for_attn,
            )

        return cell_embeddings, {"chunk_size": chunk_size, "n_chunks": len(chunks_inputs)}

    # -- HDF5 writer ---------------------------------------------------
    def _persist_attentions(
        self,
        *,
        attentions_h5_path: Path,
        chunks_outputs: list,
        sample_ids: list[int] | None,
        provenance_for_attn: pd.DataFrame | None,
    ) -> None:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "save_attentions=True requires h5py. `pip install h5py>=3.8`."
            ) from e

        n_layers = len(chunks_outputs[0].attentions)  # tuple, len = n_layers
        layers_idx = self._resolve_layer_idx(n_layers)
        # heads dimension lives at axis 1 of each attention tensor
        head_dim = chunks_outputs[0].attentions[0].shape[1]
        heads_idx = self._resolve_head_idx(head_dim)

        attentions_h5_path.parent.mkdir(parents=True, exist_ok=True)
        with __import__("h5py").File(attentions_h5_path, "a") as h5:
            h5.attrs["model_name"] = self.model_name
            h5.attrs["hidden_size"] = self.hidden_size
            h5.attrs["max_length"] = self.max_length
            h5.attrs["n_layers"] = n_layers
            h5.attrs["n_layers_kept"] = len(layers_idx)
            h5.attrs["n_heads"] = head_dim
            h5.attrs["n_heads_kept"] = len(heads_idx)
            h5.attrs["dtype"] = "fp16"
            h5.attrs["gzip_level"] = self.gzip_level
            h5.attrs["normalize"] = self.normalize_mode
            samples = h5.require_group("samples")

            for chunk_i, out in enumerate(chunks_outputs):
                # legacy chunk_size = 1, so one sample per chunk
                sample_id = (
                    int(sample_ids[chunk_i]) if sample_ids is not None else int(chunk_i)
                )

                # attn: (n_layers, batch=1, n_heads, L, L) -> (n_layers, n_heads, L, L)
                stacked = torch.stack(out.attentions, dim=0)
                attn = stacked[layers_idx][:, 0, :, :, :]  # (kept_layers, n_heads, L, L)
                attn = attn[:, heads_idx, :, :]            # (kept_layers, kept_heads, L, L)

                attn = attn.to(torch.float16)
                if self.normalize_mode == "row_softmax_preserved":
                    s = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    attn = attn / s
                elif self.normalize_mode == "none":
                    pass
                else:
                    raise ValueError(
                        f"Unsupported normalize mode: {self.normalize_mode!r}; "
                        f"use 'row_softmax_preserved' or 'none'."
                    )

                key = str(sample_id)
                if key in samples:
                    del samples[key]  # idempotent overwrite

                shape = tuple(attn.shape)
                chunks = (1, 1, shape[-2], shape[-1])
                ds = samples.create_dataset(
                    key,
                    data=attn.cpu().numpy(),
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=self.gzip_level,
                )
                ds.attrs["sample_id"] = sample_id
                if provenance_for_attn is not None:
                    row = provenance_for_attn.iloc[chunk_i]
                    for col in ("row_in_table", "row_idx", "missing_kind", "special_token"):
                        if col in provenance_for_attn.columns:
                            v = row[col]
                            ds.attrs[col] = (
                                str(v) if isinstance(v, str) or pd.isna(v) or hasattr(v, "tolist") is False
                                else v
                            )


# ---------------------------------------------------------------------------
# MiniLM peer embedder + the shared meanpool operation.
# ---------------------------------------------------------------------------
MINILM_DIM: int = 384


def _l2_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (zero rows map to themselves)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return (x / norms).astype(np.float32)


def meanpool_embeddings(
    context_emb: np.ndarray,
    target_name_emb: np.ndarray,
    *,
    hidden_size: int,
) -> np.ndarray:
    """L2-normalized 50/50 mean-pool of a context embedding with the
    MiniLM embedding of the (normalized) target-column name.

        v1 = L2(context_emb)                     # dim H
        v2 = L2(MiniLM(target_col_name))         # dim 384
        meanpool = L2((v1 + v2) / 2)

    Dimension handling (spec invariant 9):
      * MiniLM runs (H == 384): plain same-dim mean, no padding.
      * TAPAS runs (H == 768/1024): zero-pad the 384-dim v2 up to H first.
    The pooled partner is ALWAYS the MiniLM target-name embedding.
    """
    if context_emb.shape[1] != hidden_size:
        raise ValueError(
            f"context_emb dim {context_emb.shape[1]} != hidden_size {hidden_size}"
        )
    if target_name_emb.shape[1] != MINILM_DIM:
        raise ValueError(
            f"target_name_emb dim {target_name_emb.shape[1]} != {MINILM_DIM}"
        )
    if context_emb.shape[0] != target_name_emb.shape[0]:
        raise ValueError("row count mismatch between context and target-name embeddings")

    v1 = _l2_rows(context_emb)
    v2 = _l2_rows(target_name_emb)
    if hidden_size != MINILM_DIM:
        padded = np.zeros((len(v2), hidden_size), dtype=np.float32)
        padded[:, :MINILM_DIM] = v2
        v2 = padded
    return _l2_rows((v1 + v2) / 2.0)


class MiniLMCellEmbedder(BaseEmbedder):
    """Peer encoder using all-MiniLM-L6-v2 over serialized sentences.

    `[EMPTY]` and `[?]` are added as ATOMIC tokens and the embedding matrix
    is resized once, so MiniLM has explicit support for the empty / mask
    routing that TAPAS gets natively.
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        n_added = self.model.tokenizer.add_tokens(["[EMPTY]", "[?]"])
        if n_added:
            self.model._first_module().auto_model.resize_token_embeddings(
                len(self.model.tokenizer)
            )
        for tok in ("[EMPTY]", "[?]"):
            ids = self.model.tokenizer(tok, add_special_tokens=False)["input_ids"]
            atomic = "atomic" if len(ids) == 1 else "NOT atomic"
            print(f"[minilm] {tok!r} -> {ids}  ({atomic})")

        self.model_name = model_name
        self.normalize = normalize
        self.hidden_size = int(self.model.get_sentence_embedding_dimension())

    def encode_sentences(self, sentences: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        ).astype(np.float32)

    def encode_target_names(self, names: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode (already-normalized) target-column names; always L2."""
        return self.model.encode(
            names,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_table(self, **kwargs):  # pragma: no cover - not used for MiniLM
        raise NotImplementedError(
            "MiniLMCellEmbedder uses encode_sentences(), not encode_table()."
        )


# A single MiniLM instance is reused as the meanpool partner for ALL
# encoders (the pooled v2 is always MiniLM(target_col name)).
_MINILM_NAME_ENCODER: MiniLMCellEmbedder | None = None


def get_minilm_name_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MiniLMCellEmbedder:
    """Lazily build + cache the MiniLM encoder used for target-name pooling."""
    global _MINILM_NAME_ENCODER
    if _MINILM_NAME_ENCODER is None:
        _MINILM_NAME_ENCODER = MiniLMCellEmbedder(model_name=model_name)
    return _MINILM_NAME_ENCODER
