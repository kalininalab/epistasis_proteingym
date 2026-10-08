#!/usr/bin/env python
"""Conditional variant-effect scoring with the official VESPA and VESPAl heads."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices

from score_prott5 import AA, INPUT, infer_wt, load_model, mutate, parse_double

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
VESPA_ROOT = Path("/data/users/akolchina/software/VESPA")
AA_TO_I = {aa: i for i, aa in enumerate(AA)}


class ConservationCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(32, 9, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x):
        return self.classifier(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)


def load_conservation(device):
    model = ConservationCNN().to(device)
    checkpoint = torch.load(VESPA_ROOT / "models/ProtT5cons_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def load_heads():
    result = {}
    for key, filename in {
        "VESPA": "VESPA-10LR_Cons_Blsm_Prob.pkl",
        "VESPAl": "VESPAl-10LR_Cons_Blsm.pkl",
    }.items():
        with open(VESPA_ROOT / "models" / filename, "rb") as handle:
            result[key] = pickle.load(handle)
    return result


def encode_batch(sequences, tokenizer, model, conservation, device):
    texts = [" ".join(s) for s in sequences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = model.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        cons = torch.softmax(conservation(hidden[:, : max(map(len, sequences))].float()), dim=1)
    return [cons[k, :, : len(seq)].T.cpu().numpy() for k, seq in enumerate(sequences)]


def masked_batch(requests, tokenizer, device):
    texts = []
    for sequence, pos in requests:
        chars = list(sequence)
        chars[pos] = "<extra_id_0>"
        texts.append(" ".join(chars))
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)


def requested_log_odds(requests, tokenizer, model, device, batch_size):
    unique = list(dict.fromkeys(requests))
    aa_ids = torch.tensor([tokenizer.get_vocab()["▁" + aa] for aa in AA], device=device)
    result = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        enc = masked_batch(batch, tokenizer, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**enc, labels=enc.input_ids).logits
        for row, key in enumerate(batch):
            result[key] = torch.log_softmax(logits[row, key[1], aa_ids].float(), -1).cpu().numpy()
    return result


def predict_heads(requests, sequences, conservations, log_odds, heads):
    blosum = substitution_matrices.load("BLOSUM62")
    features = {"VESPA": [], "VESPAl": []}
    for background, pos, old, new in requests:
        cons = conservations[background][pos]
        b = float(blosum[old, new])
        lo = float(log_odds[(background, pos)][AA_TO_I[new]] - log_odds[(background, pos)][AA_TO_I[old]])
        features["VESPAl"].append(np.r_[cons, b])
        features["VESPA"].append(np.r_[cons, b, lo])
    output = {}
    for name in ("VESPA", "VESPAl"):
        x = np.asarray(features[name])
        probabilities = np.mean([est.predict_proba(x)[:, 1] for est in heads[name].values()], axis=0)
        # ProteinGym orients these as fitness-like scores: minus predicted disease probability.
        output[name] = -probabilities
    return output


def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--revision", default="973be27c52ee6474de9c945952a8008aeb2a1a73")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-doubles", type=int)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "scores.csv").exists() and (args.output / "runtime.json").exists():
        print(f"complete: VESPA/{args.dataset}", flush=True)
        return

    data = pd.read_csv(INPUT / f"{args.dataset}.csv", low_memory=False)
    doubles = data[data.num_mutations.eq(2)].copy()
    if args.max_doubles and len(doubles) > args.max_doubles:
        rng = np.random.default_rng(args.seed)
        doubles = doubles.iloc[np.sort(rng.choice(len(doubles), args.max_doubles, replace=False))].copy()
    wt = infer_wt(doubles)
    records, backgrounds, requests = [], {wt: None}, []
    for row in doubles.itertuples():
        ma, mb = parse_double(row.mutant)
        sa, sb = mutate(wt, ma), mutate(wt, mb)
        sab = mutate(sa, mb)
        if sab != row.mutated_sequence:
            raise ValueError(f"AB mismatch: {row.mutant}")
        backgrounds.update({sa: None, sb: None})
        req = ((wt, ma[1], ma[0], ma[2]), (sb, ma[1], ma[0], ma[2]),
               (wt, mb[1], mb[0], mb[2]), (sa, mb[1], mb[0], mb[2]))
        requests.extend(req)
        records.append((row, req))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required")
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint, revision=args.revision, local_files_only=True, do_lower_case=False)
    model = load_model(args.checkpoint, args.revision).to(device).eval()
    conservation = load_conservation(device)
    heads = load_heads()

    sequence_list = list(backgrounds)
    cons_by_sequence = {}
    for start in range(0, len(sequence_list), args.batch_size):
        chunk = sequence_list[start : start + args.batch_size]
        values = encode_batch(chunk, tokenizer, model, conservation, device)
        cons_by_sequence.update(zip(chunk, values))
    position_requests = [(s, pos) for s, pos, _, _ in requests]
    log_odds = requested_log_odds(position_requests, tokenizer, model, device, args.batch_size)
    unique_requests = list(dict.fromkeys(requests))
    predictions = predict_heads(unique_requests, backgrounds, cons_by_sequence, log_odds, heads)
    lookup = {name: dict(zip(unique_requests, values)) for name, values in predictions.items()}

    rows = []
    for row, req in records:
        base = dict(dataset=args.dataset, mutant=row.mutant, mutated_sequence=row.mutated_sequence,
                    pair_name=row.pair_name, experimental_dG=row.dG,
                    experimental_coupling=row.thermodynamic_coupling, epistatic=row.epistatic)
        for name in ("VESPA", "VESPAl"):
            da0, da_b, db0, db_a = [lookup[name][r] for r in req]
            eba, eab = da_b - da0, db_a - db0
            econd = 0.5 * (eba + eab)
            direct = 0.5 * ((da0 + db_a) + (db0 + da_b))
            stored_default = float(getattr(row, name))
            base.update({f"{name}_default": stored_default, f"{name}_A_WT": da0,
                         f"{name}_A_B": da_b, f"{name}_B_WT": db0, f"{name}_B_A": db_a,
                         f"{name}_E_B_to_A": eba, f"{name}_E_A_to_B": eab,
                         f"{name}_E_cond": econd,
                         f"{name}_S_cond_direct": direct,
                         f"{name}_S_cond": stored_default + econd,
                         f"{name}_recomputed_default": da0 + db0})
        rows.append(base)
    output = pd.DataFrame(rows)
    atomic_csv(output, args.output / "scores.csv")
    runtime = dict(dataset=args.dataset, models=["VESPA", "VESPAl"], n_double_mutants=len(output),
                   n_background_sequences=len(backgrounds), n_unique_requests=len(unique_requests),
                   total_seconds=time.perf_counter() - started, device=str(device))
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
