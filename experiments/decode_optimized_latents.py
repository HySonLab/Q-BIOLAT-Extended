import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, hidden: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mask = nn.Linear(hidden, seq_len)
        self.aa = nn.Linear(hidden, seq_len * 20)

    def forward(self, z):
        h = self.shared(z)
        mask_logits = self.mask(h)                     # [B, L]
        aa_logits = self.aa(h).view(-1, self.seq_len, 20)  # [B, L, 20]
        return mask_logits, aa_logits


def load_decoder_checkpoint(path: str, device: str = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=device)
    required = ["decoder_state_dict", "wt_seq", "seq_len", "config"]
    for key in required:
        if key not in ckpt:
            raise ValueError(f"{path} does not contain required key '{key}'")
    return ckpt


def build_decoder_from_checkpoint(ckpt: Dict, device: str = "cpu") -> nn.Module:
    cfg = ckpt["config"]
    latent_dim = int(cfg["latent_dim"])
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("decoder_hidden_dim", 256)))
    seq_len = int(ckpt["seq_len"])

    model = Decoder(latent_dim=latent_dim, seq_len=seq_len, hidden=hidden_dim).to(device)
    model.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()
    return model


def load_codes_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    for key in ["x", "X", "z_bin", "bits", "codes"]:
        if key in data:
            arr = data[key]
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr.astype(np.float32)
    raise ValueError(f"Could not find code array in npz file: {path}")


def load_codes_csv(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    bit_cols = [c for c in df.columns if c.startswith("bit_")]
    if not bit_cols:
        raise ValueError(f"No bit_* columns found in csv: {path}")
    arr = df[bit_cols].values.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_codes_json(path: str, expected_dim: Optional[int] = None, method: Optional[str] = None) -> np.ndarray:
    """
    Supports current multiseed JSON format with:
      - top-level decoder_codes[method] = list of codes
      - fallback raw_runs[*].methods[method].best_code
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = []

    # Preferred format
    if "decoder_codes" in data:
        if method is None:
            for _, arr in data["decoder_codes"].items():
                codes.extend(arr)
        else:
            if method not in data["decoder_codes"]:
                raise ValueError(f"Method '{method}' not found in decoder_codes of {path}")
            codes = data["decoder_codes"][method]

    # Fallback format
    elif "raw_runs" in data:
        for run in data["raw_runs"]:
            methods = run.get("methods", {})
            if method is None:
                for _, m in methods.items():
                    if "best_code" in m:
                        codes.append(m["best_code"])
            else:
                if method in methods and "best_code" in methods[method]:
                    codes.append(methods[method]["best_code"])

    if not codes:
        raise ValueError(f"No optimized codes found in json: {path}")

    arr = np.array(codes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"Expected latent dim {expected_dim}, but got {arr.shape[1]} from {path}"
        )

    return arr


def load_optimized_codes(path: str, expected_dim: Optional[int] = None, method: Optional[str] = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_codes_npz(path)
    if ext == ".csv":
        return load_codes_csv(path)
    if ext == ".json":
        return load_codes_json(path, expected_dim=expected_dim, method=method)
    raise ValueError(f"Unsupported file extension for optimized codes: {path}")


def decode_sequences(
    model: nn.Module,
    codes: np.ndarray,
    wt_seq: str,
    mask_threshold: float = 0.5,
    device: str = "cpu",
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    z = torch.from_numpy(codes.astype(np.float32)).to(device)

    with torch.no_grad():
        mask_logits, aa_logits = model(z)
        mask_prob = torch.sigmoid(mask_logits).cpu().numpy()   # [N, L]
        aa_pred = aa_logits.argmax(dim=-1).cpu().numpy()       # [N, L]

    decoded = []
    for i in range(codes.shape[0]):
        seq = list(wt_seq)
        for pos in range(len(wt_seq)):
            if mask_prob[i, pos] > mask_threshold:
                seq[pos] = AMINO_ACIDS[int(aa_pred[i, pos])]
        decoded.append("".join(seq))

    return decoded, mask_prob, aa_pred


def deduplicate_sequences(seqs: List[str], codes: np.ndarray, mask_prob: Optional[np.ndarray] = None):
    seen = {}
    uniq_seqs = []
    uniq_codes = []
    uniq_mask_prob = []

    for i, (s, c) in enumerate(zip(seqs, codes)):
        if s not in seen:
            seen[s] = True
            uniq_seqs.append(s)
            uniq_codes.append(c)
            if mask_prob is not None:
                uniq_mask_prob.append(mask_prob[i])

    uniq_codes = np.array(uniq_codes, dtype=np.float32)
    uniq_mask_prob = np.array(uniq_mask_prob, dtype=np.float32) if mask_prob is not None else None
    return uniq_seqs, uniq_codes, uniq_mask_prob


def load_oracle_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_sequences_esm(
    sequences: List[str],
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 8,
    device: str = "cpu",
):
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            h = out.last_hidden_state
            attn = toks["attention_mask"]

            for i in range(h.shape[0]):
                valid = attn[i].bool()
                vecs = h[i][valid]

                # Drop BOS/EOS if available
                if vecs.shape[0] >= 3:
                    vecs = vecs[1:-1]

                emb = vecs.mean(dim=0)
                all_embeddings.append(emb.cpu().numpy())

    return np.stack(all_embeddings, axis=0).astype(np.float32)


def score_with_oracle(
    sequences: List[str],
    oracle_model,
    esm_model_name: str,
    embed_batch_size: int,
    device: str,
) -> np.ndarray:
    X = embed_sequences_esm(
        sequences=sequences,
        model_name=esm_model_name,
        batch_size=embed_batch_size,
        device=device,
    )
    scores = oracle_model.predict(X)
    return np.asarray(scores, dtype=np.float32)


def save_results_csv(
    path: str,
    sequences: List[str],
    scores: Optional[np.ndarray] = None,
    mask_prob: Optional[np.ndarray] = None,
):
    fieldnames = ["rank", "sequence"]
    if scores is not None:
        fieldnames.append("oracle_score")
    if mask_prob is not None:
        fieldnames.append("mean_mutation_prob")

    rows = []
    for i, seq in enumerate(sequences):
        row = {"rank": i + 1, "sequence": seq}
        if scores is not None:
            row["oracle_score"] = float(scores[i])
        if mask_prob is not None:
            row["mean_mutation_prob"] = float(mask_prob[i].mean())
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized-codes", type=str, required=True,
                        help="Path to optimized codes (.json/.npz/.csv)")
    parser.add_argument("--decoder-ckpt", type=str, required=True,
                        help="Path to trained decoder checkpoint (.pt)")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--method", type=str, default=None,
                        choices=[None, "simulated_annealing", "genetic_algorithm",
                                 "random_search", "greedy_hill_climb", "latent_bo"])
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--deduplicate", action="store_true")

    # Optional oracle scoring
    parser.add_argument("--oracle-model", type=str, default=None)
    parser.add_argument("--esm-model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--embed-batch-size", type=int, default=8)

    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    codes = load_optimized_codes(
        args.optimized_codes,
        expected_dim=args.latent_dim,
        method=args.method,
    )

    ckpt = load_decoder_checkpoint(args.decoder_ckpt, device=args.device)
    decoder = build_decoder_from_checkpoint(ckpt, device=args.device)
    wt_seq = ckpt["wt_seq"]

    decoded_sequences, mask_prob, _ = decode_sequences(
        model=decoder,
        codes=codes,
        wt_seq=wt_seq,
        mask_threshold=args.mask_threshold,
        device=args.device,
    )

    if args.deduplicate:
        decoded_sequences, codes, mask_prob = deduplicate_sequences(
            decoded_sequences, codes, mask_prob
        )

    oracle_scores = None
    if args.oracle_model is not None:
        oracle = load_oracle_model(args.oracle_model)
        oracle_scores = score_with_oracle(
            sequences=decoded_sequences,
            oracle_model=oracle,
            esm_model_name=args.esm_model_name,
            embed_batch_size=args.embed_batch_size,
            device=args.device,
        )

        order = np.argsort(-oracle_scores)
        decoded_sequences = [decoded_sequences[i] for i in order]
        oracle_scores = oracle_scores[order]
        mask_prob = mask_prob[order] if mask_prob is not None else None

    save_results_csv(
        path=args.output_csv,
        sequences=decoded_sequences,
        scores=oracle_scores,
        mask_prob=mask_prob,
    )

    summary = {
        "method": args.method,
        "n_input_codes": int(len(codes)),
        "n_decoded_sequences": int(len(decoded_sequences)),
        "oracle_scored": args.oracle_model is not None,
    }

    if oracle_scores is not None and len(oracle_scores) > 0:
        summary["best_oracle_score"] = float(np.max(oracle_scores))
        summary["top_10_mean"] = float(
            np.mean(np.sort(oracle_scores)[-min(10, len(oracle_scores)):])
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved decoded sequences to: {args.output_csv}")


if __name__ == "__main__":
    main()
