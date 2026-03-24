import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_latent_npz(path):
    data = np.load(path, allow_pickle=True)

    # Support several possible latent keys
    if "X" in data:
        Z = data["X"]
    elif "z_cont" in data:
        Z = data["z_cont"]
    elif "embeddings" in data:
        Z = data["embeddings"]
    else:
        raise ValueError(f"Cannot find latent array in npz: {path}")

    if "items" not in data:
        raise ValueError(f"Cannot find 'items' in npz: {path}")

    items = data["items"]
    return Z.astype(np.float32), items


def infer_wt(sequences):
    L = len(sequences[0])
    wt = []
    for i in range(L):
        counts = {}
        for s in sequences:
            counts[s[i]] = counts.get(s[i], 0) + 1
        wt.append(max(counts.items(), key=lambda x: x[1])[0])
    return "".join(wt)


def build_targets(sequences, wt):
    N = len(sequences)
    L = len(wt)

    mut = np.zeros((N, L), dtype=np.float32)
    aa = np.full((N, L), -100, dtype=np.int64)

    for i, s in enumerate(sequences):
        if len(s) != L:
            raise ValueError("Sequence length mismatch")
        for j, (w, c) in enumerate(zip(wt, s)):
            if c != w:
                mut[i, j] = 1.0
                if c not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid '{c}'")
                aa[i, j] = AA_TO_IDX[c]

    return mut, aa


class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, hidden=256):
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


def compute_metrics(mask_true, aa_true, mask_logits, aa_logits):
    mask_prob = torch.sigmoid(mask_logits)
    mask_pred = (mask_prob > 0.5).float()

    tp = ((mask_true == 1) & (mask_pred == 1)).sum().item()
    fp = ((mask_true == 0) & (mask_pred == 1)).sum().item()
    fn = ((mask_true == 1) & (mask_pred == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    aa_pred = aa_logits.argmax(dim=-1)
    valid = aa_true != -100
    if valid.sum() > 0:
        aa_acc = (aa_pred[valid] == aa_true[valid]).float().mean().item()
    else:
        aa_acc = 0.0

    pos_acc = (mask_pred == mask_true).float().mean().item()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aa_acc": aa_acc,
        "pos_acc": pos_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-npz", required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mask-loss-weight", type=float, default=2.0)
    parser.add_argument("--aa-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    Z, items = load_latent_npz(args.latent_npz)
    sequences = [str(s) for s in items.tolist()]
    wt = infer_wt(sequences)

    Z_train, Z_test, seq_train, seq_test = train_test_split(
        Z, sequences, test_size=0.2, random_state=args.seed
    )

    mut_train, aa_train = build_targets(seq_train, wt)
    mut_test, aa_test = build_targets(seq_test, wt)

    device = "cpu"
    model = Decoder(args.latent_dim, len(wt), hidden=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pos = mut_train.sum()
    neg = mut_train.size - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)

    history = []

    for epoch in range(args.epochs):
        model.train()

        idx = np.random.permutation(len(Z_train))
        for i in range(0, len(idx), args.batch_size):
            batch = idx[i:i + args.batch_size]

            z = torch.from_numpy(Z_train[batch]).to(device)
            m = torch.from_numpy(mut_train[batch]).to(device)
            a = torch.from_numpy(aa_train[batch]).to(device)

            mask_logits, aa_logits = model(z)

            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits, m, pos_weight=pos_weight
            )

            aa_loss = F.cross_entropy(
                aa_logits.reshape(-1, 20),
                a.reshape(-1),
                ignore_index=-100,
            )

            loss = args.mask_loss_weight * mask_loss + args.aa_loss_weight * aa_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z = torch.from_numpy(Z_test).to(device)
                m = torch.from_numpy(mut_test).to(device)
                a = torch.from_numpy(aa_test).to(device)

                mask_logits, aa_logits = model(z)
                metrics = compute_metrics(m, a, mask_logits, aa_logits)

            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "aa_acc": metrics["aa_acc"],
                "pos_acc": metrics["pos_acc"],
            })

            print(
                f"epoch {epoch}, loss {loss.item():.4f}, "
                f"f1 {metrics['f1']:.4f}, aa_acc {metrics['aa_acc']:.4f}, "
                f"pos_acc {metrics['pos_acc']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(Z_test).to(device)
        m = torch.from_numpy(mut_test).to(device)
        a = torch.from_numpy(aa_test).to(device)

        mask_logits, aa_logits = model(z)
        metrics = compute_metrics(m, a, mask_logits, aa_logits)

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    result = {
        "latent_npz": args.latent_npz,
        "latent_dim": args.latent_dim,
        "wt_length": len(wt),
        "train_size": int(len(Z_train)),
        "test_size": int(len(Z_test)),
        "metrics": metrics,
        "history": history,
    }

    with open(args.output_prefix + ".json", "w") as f:
        json.dump(result, f, indent=2)

    print(result["metrics"])


if __name__ == "__main__":
    main()
