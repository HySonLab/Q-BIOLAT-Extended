import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def read_csv_dataset(path: str, sequence_col: str, fitness_col: str):
    df = pd.read_csv(path)
    if sequence_col not in df.columns:
        raise ValueError(f"Missing sequence column: {sequence_col}")
    if fitness_col not in df.columns:
        raise ValueError(f"Missing fitness column: {fitness_col}")

    sequences = df[sequence_col].astype(str).str.upper().tolist()
    fitness = df[fitness_col].astype(float).to_numpy(dtype=np.float64)
    return sequences, fitness, df


def clean_sequence(seq: str) -> str:
    # Keep only alphabetic characters, uppercase
    seq = "".join(ch for ch in seq.upper() if ch.isalpha())
    if len(seq) == 0:
        raise ValueError("Encountered empty sequence after cleaning.")
    return seq


def sanitize_sequences(sequences):
    cleaned = []
    for s in sequences:
        cleaned.append(clean_sequence(s))
    return cleaned


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: (B, L, D)
    # attention_mask:    (B, L)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)                          # (B, 1)
    return summed / counts


def embed_sequences(
    sequences,
    model_name: str,
    batch_size: int = 8,
    max_length: int = 1024,
    device: str = "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]

            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)

            # Use final hidden states from the encoder
            pooled = mean_pool_last_hidden(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def random_project(X: np.ndarray, out_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim))
    Z = X @ W
    return Z, W


def binarize_embeddings(Z: np.ndarray, method: str = "median"):
    if method == "median":
        thresholds = np.median(Z, axis=0, keepdims=True)
        X_bin = (Z > thresholds).astype(np.int32)
    elif method == "sign":
        thresholds = np.zeros((1, Z.shape[1]), dtype=Z.dtype)
        X_bin = (Z > 0.0).astype(np.int32)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    return X_bin, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-npz", type=str, default="artifacts/real_peptides_esm_binary.npz")

    parser.add_argument("--sequence-col", type=str, default="sequence")
    parser.add_argument("--fitness-col", type=str, default="fitness")

    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Any Hugging Face ESM/ESM-2 checkpoint."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--binarize", type=str, choices=["median", "sign"], default="median")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)

    print(f"Reading CSV: {args.input_csv}")
    sequences, fitness, df = read_csv_dataset(
        args.input_csv,
        sequence_col=args.sequence_col,
        fitness_col=args.fitness_col,
    )
    sequences = sanitize_sequences(sequences)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Embedding with model: {args.model_name}")
    print(f"Device: {args.device}")

    embeddings = embed_sequences(
        sequences,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    print("Dense embedding shape:", embeddings.shape)

    if args.project_dim > 0 and args.project_dim < embeddings.shape[1]:
        Z, projection = random_project(embeddings, out_dim=args.project_dim, seed=args.seed)
        print("Projected embedding shape:", Z.shape)
    else:
        Z = embeddings
        projection = None
        print("Skipping projection; using full embedding dimension.")

    X_bin, thresholds = binarize_embeddings(Z, method=args.binarize)

    print("Binary latent shape:", X_bin.shape)
    print("Mean bit activation:", float(X_bin.mean()))

    np.savez(
        args.output_npz,
        X=X_bin.astype(np.int32),
        y=fitness.astype(np.float64),
        items=np.array(sequences, dtype=object),
        embeddings=embeddings.astype(np.float32),
        projected=Z.astype(np.float32),
        thresholds=thresholds.astype(np.float32),
        model_name=np.array(args.model_name, dtype=object),
        sequence_col=np.array(args.sequence_col, dtype=object),
        fitness_col=np.array(args.fitness_col, dtype=object),
        projection=None if projection is None else projection.astype(np.float32),
    )

    print(f"Saved dataset to: {args.output_npz}")


if __name__ == "__main__":
    main()
