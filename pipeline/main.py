"""Pipeline orchestrator (refactor).

Encoder-agnostic runner. Loads `samples/sample_manifest.parquet`, then for
every selected (encoder, ablation) pair iterates over (file, target_col)
groups, prepares inputs (TAPAS table slice or MiniLM sentence), embeds,
writes one parquet shard per group, and lazy-concats shards into the final
`prediction_data/<encoder>/<ablation>/pred_dataset.parquet`.

The run is the cartesian product {encoders} x {ablations} from config.yaml
(co_missing_drop is just one ablation entry). Restrict on the CLI with
`--encoder MiniLM --ablation co_missing_drop`, or run everything with
`--all`.

Crash-resumable: per-shard `progress.json` records state + manifest
sha256. Re-running the same (encoder, ablation) skips groups already done;
refuses to resume if the manifest changed.

Usage:
    python pipeline/main.py --all
    python pipeline/main.py --encoder MiniLM --ablation default
    python pipeline/main.py --encoder TAPAS-base --all-ablations
    python pipeline/main.py --all --smoke-limit-groups 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import yaml

from _pipeline_utils import (
    PIPELINE_DIR,
    REPO_ROOT,
    atomic_write_json,
    git_sha,
    load_sibling_module,
    now_iso,
    sha256,
)


cleaning = load_sibling_module("01_table_cleaning.py", alias="cleaning")
input_prep = load_sibling_module("02_input_prep.py", alias="input_prep")
build_embeddings = load_sibling_module("04_build_embeddings.py", alias="build_embeddings")
build_pred_dataset_mod = load_sibling_module("05_build_pred_dataset.py", alias="build_pred_dataset")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderConfig:
    id: str
    name: str
    family: str
    hidden_size: int
    select_one_column: bool = False
    max_length: int = 512
    queries: str = ""


@dataclass(frozen=True)
class AblationConfig:
    id: str
    header: str           # preserve | strip | only_target
    target: str           # mask | remove | meanpool
    co_missing_drop: bool


def load_config(path: Path) -> tuple[dict[str, EncoderConfig], dict[str, AblationConfig], dict]:
    cfg = yaml.safe_load(path.read_text())

    if not cfg.get("encoders"):
        raise RuntimeError(f"config.yaml at {path} has no 'encoders' list")
    encoders: dict[str, EncoderConfig] = {}
    for e in cfg["encoders"]:
        if e["id"] in encoders:
            raise RuntimeError(f"duplicate encoder id: {e['id']}")
        encoders[e["id"]] = EncoderConfig(
            id=e["id"], name=e["name"], family=e["family"],
            hidden_size=int(e["hidden_size"]),
            select_one_column=bool(e.get("select_one_column", False)),
            max_length=int(e.get("max_length", 512)),
            queries=str(e.get("queries", "")),
        )

    if not cfg.get("ablations"):
        raise RuntimeError(f"config.yaml at {path} has no 'ablations' list")
    ablations: dict[str, AblationConfig] = {}
    for a in cfg["ablations"]:
        if a["id"] in ablations:
            raise RuntimeError(f"duplicate ablation id: {a['id']}")
        ablations[a["id"]] = AblationConfig(
            id=a["id"],
            header=str(a["header"]),
            target=str(a["target"]),
            co_missing_drop=bool(a["co_missing_drop"]),
        )
    return encoders, ablations, cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {"done": [], "skipped": [], "next_shard_idx": 0}
    return json.loads(progress_path.read_text())


def _resolve_source_path(file_field: str) -> Path:
    p = Path(file_field)
    return p if p.is_absolute() else REPO_ROOT / p


def _make_embedder(encoder: EncoderConfig, attentions_cfg: dict | None):
    if encoder.family == "tapas":
        emb = build_embeddings.TapasCellEmbedder(
            model_name=encoder.name,
            max_length=encoder.max_length,
            select_one_column=encoder.select_one_column,
            queries=encoder.queries,
            attention_layers_to_keep=(
                attentions_cfg.get("layers_to_keep") if attentions_cfg else None
            ),
            attention_heads_to_keep=(
                attentions_cfg.get("heads_to_keep") if attentions_cfg else None
            ),
            normalize=(
                attentions_cfg.get("normalize", "row_softmax_preserved")
                if attentions_cfg else "row_softmax_preserved"
            ),
            gzip_level=int(attentions_cfg.get("gzip_level", 4)) if attentions_cfg else 4,
        )
        if emb.hidden_size != encoder.hidden_size:
            raise RuntimeError(
                f"{encoder.name} reports hidden_size={emb.hidden_size}, "
                f"config says {encoder.hidden_size}. Fix config.yaml."
            )
        return emb
    if encoder.family == "minilm":
        emb = build_embeddings.MiniLMCellEmbedder(model_name=encoder.name)
        if emb.hidden_size != encoder.hidden_size:
            raise RuntimeError(
                f"{encoder.name} reports hidden_size={emb.hidden_size}, "
                f"config says {encoder.hidden_size}. Fix config.yaml."
            )
        return emb
    raise ValueError(f"unknown encoder family: {encoder.family!r}")


def _embed_group(
    *,
    encoder: EncoderConfig,
    ablation: AblationConfig,
    embedder,
    df: pd.DataFrame,
    slice_: pd.DataFrame,
    save_attentions: bool,
    h5_path: Path | None,
):
    """Return an (n_rows, hidden_size) ndarray + provenance for one group."""
    import numpy as np

    abl_spec = input_prep.AblationSpec(
        id=ablation.id, header=ablation.header,
        target=ablation.target, co_missing_drop=ablation.co_missing_drop,
    )

    if encoder.family == "tapas":
        prepared = input_prep.prepare_inputs(
            family="tapas", df=df, manifest_slice=slice_, ablation=abl_spec,
            normalize_fn=cleaning._normalize,
            normalize_columns_fn=cleaning._normalize_columns,
            canonicalize_missing_fn=input_prep.canonicalize_missing,
        )
        cells, _ = embedder.encode_table(
            masked=prepared.masked,
            target_col_idx=prepared.target_col_idx,
            sample_ids=slice_["sample_id"].tolist(),
            save_attentions=save_attentions,
            attentions_h5_path=h5_path,
            provenance_for_attn=prepared.provenance,
        )
        context_emb = cells.detach().cpu().numpy().astype(np.float32)
    else:  # minilm
        prepared = input_prep.prepare_inputs(
            family="minilm", df=df, manifest_slice=slice_, ablation=abl_spec,
            normalize_fn=cleaning._normalize,
            is_missing_fn=cleaning.is_missing,
            mask_token=input_prep.MASK_TOKEN,
            empty_token=input_prep.EMPTY_TOKEN,
            format_cell_fn=cleaning.format_cell_value,
        )
        context_emb = embedder.encode_sentences(prepared.sentences)

    if ablation.target == "meanpool":
        name_encoder = build_embeddings.get_minilm_name_encoder()
        name_emb = name_encoder.encode_target_names([prepared.target_col_name])
        name_emb = np.repeat(name_emb, len(context_emb), axis=0)
        emb = build_embeddings.meanpool_embeddings(
            context_emb, name_emb, hidden_size=encoder.hidden_size
        )
    else:
        emb = context_emb

    return emb, prepared.provenance


def run_one(
    *,
    encoder: EncoderConfig,
    ablation: AblationConfig,
    embedder,
    manifest: pd.DataFrame,
    manifest_path: Path,
    manifest_sha: str,
    out_dir: Path,
    save_attentions: bool = False,
    smoke_limit_groups: int | None = None,
    verbose: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(exist_ok=True)
    progress_path = out_dir / "progress.json"
    progress = _load_progress(progress_path)

    if progress.get("manifest_sha256") and progress["manifest_sha256"] != manifest_sha:
        raise RuntimeError(
            f"Manifest sha256 mismatch for {encoder.id}/{ablation.id}.\n"
            f"  progress.json sha = {progress['manifest_sha256']}\n"
            f"  current  sha     = {manifest_sha}\n"
            f"Sampling changed since the last shard. Rerun 00a and delete this "
            f"run's _shards/, or point --manifest at the older manifest."
        )
    progress["manifest_sha256"] = manifest_sha
    progress["manifest_path"] = str(manifest_path)
    progress["encoder"] = {"id": encoder.id, "name": encoder.name, "family": encoder.family}
    progress["ablation"] = asdict(ablation)

    # TAPAS-only attention persistence.
    save_attn = save_attentions and encoder.family == "tapas"
    h5_path = out_dir / "attentions.h5" if save_attn else None

    # Group by physical column position too: after collision-renaming,
    # several distinct source columns can share one target_col label
    # (e.g. index level 10/11/12 -> "index level"). target_col_idx keeps
    # them as separate encode groups so each is embedded against the
    # correct column.
    grp_cols = ["file", "target_col", "target_col_idx"]
    grouped = manifest.groupby(grp_cols, sort=False)
    group_keys = sorted(grouped.groups.keys())
    if smoke_limit_groups is not None:
        group_keys = group_keys[:smoke_limit_groups]

    done_keys = {(d["file"], d["target_col"], d.get("target_col_idx"))
                 for d in progress["done"]}
    skipped_keys = {(d["file"], d["target_col"], d.get("target_col_idx"))
                    for d in progress.get("skipped", [])}
    n_attempted = 0
    n_failed = 0
    t0 = time.time()

    for (file_field, target_col, target_col_idx) in group_keys:
        gkey = (file_field, target_col, target_col_idx)
        if gkey in done_keys or gkey in skipped_keys:
            continue
        n_attempted += 1
        try:
            slice_ = grouped.get_group(gkey).copy()
            slice_ = slice_.sort_values("sample_id").reset_index(drop=True)
            column_type = slice_["column_type"].iloc[0]

            df = pd.read_parquet(_resolve_source_path(file_field))

            emb, provenance = _embed_group(
                encoder=encoder, ablation=ablation, embedder=embedder,
                df=df, slice_=slice_, save_attentions=save_attn, h5_path=h5_path,
            )

            shard_df = build_pred_dataset_mod.build_pred_dataset(
                cell_embeddings=emb,
                provenance=provenance,
                table_id=Path(file_field).stem,
                column_type=column_type,
                hidden_size=encoder.hidden_size,
            )

            shard_idx = progress["next_shard_idx"]
            shard_path = shard_dir / f"{shard_idx:06d}.parquet"
            tmp = shard_path.with_suffix(".parquet.tmp")
            shard_df.to_parquet(tmp, index=False)
            tmp.replace(shard_path)

            progress["done"].append({
                "file": file_field, "target_col": target_col,
                "target_col_idx": int(target_col_idx),
                "shard": shard_path.name, "n_rows": int(len(shard_df)),
            })
            progress["next_shard_idx"] = shard_idx + 1
            progress["last_checkpoint_at"] = now_iso()
            atomic_write_json(progress_path, progress)

            if verbose:
                dt = time.time() - t0
                rate = n_attempted / max(dt, 1e-6)
                print(f"[{encoder.id}/{ablation.id}] ok {file_field} / {target_col}  "
                      f"shard={shard_path.name}  ({n_attempted}/{len(group_keys)}, "
                      f"{rate:.2f} grp/s)")
        except Exception as e:
            n_failed += 1
            progress.setdefault("skipped", []).append({
                "file": file_field, "target_col": target_col,
                "target_col_idx": int(target_col_idx),
                "reason": f"{type(e).__name__}: {e}",
            })
            atomic_write_json(progress_path, progress)
            print(f"[{encoder.id}/{ablation.id}] FAIL {file_field} / {target_col}: {e}",
                  file=sys.stderr)

    pred_path = _concat_shards(shard_dir, out_dir / "pred_dataset.parquet", progress=progress)

    n_total = sum(d["n_rows"] for d in progress["done"])
    metadata = {
        "encoder": {"id": encoder.id, "name": encoder.name, "family": encoder.family,
                    "hidden_size": encoder.hidden_size},
        "ablation": asdict(ablation),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "n_shards": len(progress["done"]),
        "n_rows_total": n_total,
        "n_groups_attempted": n_attempted,
        "n_groups_failed": n_failed,
        "save_attentions": save_attn,
        "attentions_h5": str(h5_path) if h5_path else None,
        "git_sha": git_sha(),
        "finished_at": now_iso(),
        "duration_seconds": round(time.time() - t0, 1),
        "pred_dataset": str(pred_path),
    }
    atomic_write_json(out_dir / "run_metadata.json", metadata)
    if verbose:
        print(f"[{encoder.id}/{ablation.id}] done  n_rows={n_total}  "
              f"failed={n_failed}  -> {pred_path}")
    return pred_path


def _concat_shards(shard_dir: Path, out_path: Path, progress: dict | None = None) -> Path:
    """Concat parquet shards into one pred_dataset.parquet (pandas concat
    so all-null optional-string columns promote cleanly to object dtype)."""
    if progress is not None and progress.get("done"):
        names = [d["shard"] for d in progress["done"]]
        shards = [shard_dir / n for n in names if (shard_dir / n).exists()]
    else:
        shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        pd.DataFrame().to_parquet(out_path, index=False)
        return out_path
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)
    if "special_token" in df.columns:
        df["special_token"] = df["special_token"].astype("object")
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.yaml")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Override path to sample_manifest.parquet "
                        "(default: paths.samples_dir/sample_manifest.parquet)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Override paths.prediction_data_dir")

    p.add_argument("--encoder", type=str, default=None,
                   help="Restrict to one encoder id (e.g. MiniLM).")
    p.add_argument("--ablation", type=str, default=None,
                   help="Restrict to one ablation id (e.g. co_missing_drop).")
    p.add_argument("--all-ablations", action="store_true",
                   help="Run all ablations (optionally for one --encoder).")
    p.add_argument("--all", action="store_true",
                   help="Run the full cartesian product encoders x ablations.")

    p.add_argument("--save-attentions", action="store_true",
                   help="Write attentions.h5 per TAPAS run (HDF5/fp16).")
    p.add_argument("--smoke-limit-groups", type=int, default=None,
                   help="Stop after N groups per (encoder, ablation).")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    encoders, ablations, cfg = load_config(args.config)

    paths_cfg = cfg.get("paths", {})
    attentions_cfg = cfg.get("attentions", {})

    if args.manifest is None:
        samples_dir = REPO_ROOT / paths_cfg.get("samples_dir", "samples")
        manifest_path = samples_dir / "sample_manifest.parquet"
    else:
        manifest_path = args.manifest

    out_root = (
        args.out_root if args.out_root is not None
        else REPO_ROOT / paths_cfg.get("prediction_data_dir", "prediction_data")
    )

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path} (run 00a first)")

    # ---- select encoders ----
    if args.encoder is not None:
        if args.encoder not in encoders:
            raise SystemExit(f"Unknown encoder {args.encoder!r}. Known: {list(encoders)}")
        sel_encoders = [encoders[args.encoder]]
    else:
        sel_encoders = list(encoders.values())

    # ---- select ablations ----
    if args.ablation is not None:
        if args.ablation not in ablations:
            raise SystemExit(f"Unknown ablation {args.ablation!r}. Known: {list(ablations)}")
        sel_ablations = [ablations[args.ablation]]
    else:
        sel_ablations = list(ablations.values())

    if not (args.all or args.all_ablations or args.encoder or args.ablation):
        raise SystemExit(
            "Nothing selected. Use --all, or --encoder/--ablation, or --all-ablations."
        )

    manifest = pd.read_parquet(manifest_path)
    manifest_sha = sha256(manifest_path)
    save_attentions = args.save_attentions or bool(cfg.get("defaults", {}).get("save_attentions", False))

    print(f"[main] manifest={manifest_path} ({len(manifest):,} rows)  "
          f"encoders={[e.id for e in sel_encoders]}  "
          f"ablations={[a.id for a in sel_ablations]}")

    # Outer loop on encoders so each model loads ONCE for all its ablations.
    for encoder in sel_encoders:
        print(f"[main] loading encoder {encoder.id} ({encoder.name})")
        embedder = _make_embedder(encoder, attentions_cfg)
        for ablation in sel_ablations:
            out_dir = out_root / encoder.id / ablation.id
            run_one(
                encoder=encoder,
                ablation=ablation,
                embedder=embedder,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha=manifest_sha,
                out_dir=out_dir,
                save_attentions=save_attentions,
                smoke_limit_groups=args.smoke_limit_groups,
            )


if __name__ == "__main__":
    main()
