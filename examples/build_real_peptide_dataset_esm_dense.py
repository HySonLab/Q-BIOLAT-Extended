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
    return sequences, fitness


def clean_sequence(seq: str) -> str:
    seq = "".join(ch for ch in seq.upper() if ch.isalpha())
    if len(seq) == 0:
        raise ValueError("Encountered empty sequence after cleaning.")
    return seq


def sanitize_sequences(sequences):
    return [clean_sequence(s) for s in sequences]


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
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
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]
            print(f"Embedding batch {start}:{end} / {len(sequences)}")

            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool_last_hidden(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--sequence-col", type=str, default="sequence")
    parser.add_argument("--fitness-col", type=str, default="fitness")
    parser.add_argument("--model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)

    print(f"Reading CSV: {args.input_csv}")
    sequences, fitness = read_csv_dataset(
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

    np.savez(
        args.output_npz,
        embeddings=embeddings.astype(np.float32),
        y=fitness.astype(np.float64),
        items=np.array(sequences, dtype=object),
        model_name=np.array(args.model_name, dtype=object),
        sequence_col=np.array(args.sequence_col, dtype=object),
        fitness_col=np.array(args.fitness_col, dtype=object),
    )

    print(f"Saved dense embedding dataset to: {args.output_npz}")


if __name__ == "__main__":
    main()
