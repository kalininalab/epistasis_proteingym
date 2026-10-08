#!/usr/bin/env python
"""Conditional variant-effect scoring with the official VespaG v2 ESM2 head."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from score_prott5 import AA, INPUT, infer_wt, mutate, parse_double

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
VESPAG_ROOT = Path("/data/users/akolchina/software/VespaG")
CHECKPOINT = "facebook/esm2_t36_3B_UR50D"
REVISION = "476b639933c8baad5ad09a60ac1a87f987b656fc"
AA_TO_I = {aa: i for i, aa in enumerate(AA)}


def safe_load_encoder(device):
    """Load the pinned legacy-bin checkpoint without bypassing strict key validation."""
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION, local_files_only=True)
    try:
        model = AutoModel.from_pretrained(
            CHECKPOINT, revision=REVISION, local_files_only=True, torch_dtype=torch.float16
        )
    except ValueError as exc:
        if "upgrade torch to at least v2.6" not in str(exc):
            raise
        from transformers.utils.hub import cached_file
        config = AutoConfig.from_pretrained(CHECKPOINT, revision=REVISION, local_files_only=True)
        model = AutoModel.from_config(config).half()
        index_path = cached_file(CHECKPOINT, "pytorch_model.bin.index.json", revision=REVISION, local_files_only=True)
        index = json.loads(Path(index_path).read_text())
        model_keys = set(model.state_dict())
        checkpoint_keys = {key[4:] for key in index["weight_map"] if key.startswith("esm.")}
        allowed_missing = {
            "pooler.dense.bias", "pooler.dense.weight", "rotary_embeddings.inv_freq"
        }
        missing = sorted(model_keys - checkpoint_keys - allowed_missing)
        if missing:
            raise RuntimeError(f"ESM2 encoder checkpoint is missing keys: {missing[:5]}")
        snapshot = Path(index_path).parent
        loaded = set()
        for shard in sorted(set(index["weight_map"].values())):
            state = torch.load(snapshot / shard, map_location="cpu", weights_only=True)
            encoder_state = {
                key[4:]: value for key, value in state.items()
                if key.startswith("esm.") and key[4:] in model_keys
            }
            model.load_state_dict(encoder_state, strict=False)
            loaded.update(encoder_state)
            del state
        incomplete = model_keys - loaded - allowed_missing
        if incomplete:
            raise RuntimeError(f"ESM2 encoder load incomplete: {sorted(incomplete)[:5]}")
    return tokenizer, model.to(device).eval()


def load_head(device):
    from vespag.utils import DEFAULT_MODEL_PARAMETERS, load_model
    params = dict(DEFAULT_MODEL_PARAMETERS)
    return load_model(
        **params,
        checkpoint_file=VESPAG_ROOT / "model_weights/v2/esm2.pt",
    ).to(device, dtype=torch.float).eval()


@torch.inference_mode()
def score_backgrounds(sequences, tokenizer, encoder, head, device, batch_size):
    """Return official per-background min-max-normalized Lx20 score matrices."""
    output = {}
    items = list(sequences)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = encoder(**encoded).last_hidden_state
        for row, sequence in enumerate(chunk):
            embedding = hidden[row, 1 : len(sequence) + 1].float().unsqueeze(0)
            raw = head(embedding).squeeze(0).cpu().numpy()
            denominator = float(raw.max() - raw.min())
            normalized = np.zeros_like(raw) if denominator == 0 else np.clip((raw - raw.min()) / denominator, 0, 1)
            output[sequence] = normalized
    return output


def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-doubles", type=int)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "scores.csv").exists() and (args.output / "runtime.json").exists():
        print(f"complete: VespaG/{args.dataset}", flush=True)
        return
    data = pd.read_csv(INPUT / f"{args.dataset}.csv", low_memory=False)
    doubles = data[data.num_mutations.eq(2)].copy()
    if args.max_doubles and len(doubles) > args.max_doubles:
        rng = np.random.default_rng(args.seed)
        doubles = doubles.iloc[np.sort(rng.choice(len(doubles), args.max_doubles, replace=False))].copy()
    wt = infer_wt(doubles)
    backgrounds, records = {wt: None}, []
    for row in doubles.itertuples():
        ma, mb = parse_double(row.mutant)
        sa, sb = mutate(wt, ma), mutate(wt, mb)
        if mutate(sa, mb) != row.mutated_sequence:
            raise ValueError(f"AB mismatch: {row.mutant}")
        backgrounds.update({sa: None, sb: None})
        records.append((row, ma, mb, sa, sb))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required")
    tokenizer, encoder = safe_load_encoder(device)
    head = load_head(device)
    matrices = score_backgrounds(backgrounds, tokenizer, encoder, head, device, args.batch_size)
    rows = []
    for row, ma, mb, sa, sb in records:
        da0 = float(matrices[wt][ma[1], AA_TO_I[ma[2]]])
        da_b = float(matrices[sb][ma[1], AA_TO_I[ma[2]]])
        db0 = float(matrices[wt][mb[1], AA_TO_I[mb[2]]])
        db_a = float(matrices[sa][mb[1], AA_TO_I[mb[2]]])
        path_ab, path_ba = da0 * db_a, db0 * da_b
        default = float(row.VespaG)
        direct = 0.5 * (path_ab + path_ba)
        recomputed_default = da0 * db0
        rows.append(dict(
            dataset=args.dataset, model_key="VespaG", mutant=row.mutant,
            mutated_sequence=row.mutated_sequence, pair_name=row.pair_name,
            experimental_dG=row.dG, experimental_coupling=row.thermodynamic_coupling,
            epistatic=row.epistatic, VespaG_default=default,
            VespaG_A_WT=da0, VespaG_A_B=da_b, VespaG_B_WT=db0, VespaG_B_A=db_a,
            VespaG_E_B_to_A=da_b - da0, VespaG_E_A_to_B=db_a - db0,
            VespaG_E_cond=0.5 * ((da_b - da0) + (db_a - db0)),
            VespaG_path_A_then_B=path_ab, VespaG_path_B_then_A=path_ba,
            VespaG_S_cond_direct=direct,
            VespaG_S_cond=default + (direct - recomputed_default),
            VespaG_recomputed_default=recomputed_default,
        ))
    output = pd.DataFrame(rows)
    atomic_csv(output, args.output / "scores.csv")
    runtime = dict(dataset=args.dataset, model="VespaG", encoder=CHECKPOINT, revision=REVISION,
                   n_double_mutants=len(output), n_background_sequences=len(backgrounds),
                   total_seconds=time.perf_counter() - started, device=str(device))
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
