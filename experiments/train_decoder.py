import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dense_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise ValueError(f"{path} does not contain 'embeddings'")
    if "y" not in data.files:
        raise ValueError(f"{path} does not contain 'y'")
    if "items" not in data.files:
        raise ValueError(f"{path} does not contain 'items'")

    X = data["embeddings"].astype(np.float32)
    y = data["y"].astype(np.float32)
    items = data["items"]
    return X, y, items


def load_latent_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)


class AENet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VAENet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_distribution(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu


def build_latent_encoder(model_name: str, input_dim: int, latent_dim: int, hidden_dim: int):
    if model_name == "ae":
        return AENet(input_dim, latent_dim, hidden_dim)
    if model_name == "vae":
        return VAENet(input_dim, latent_dim, hidden_dim)
    raise ValueError(f"Unknown model_name: {model_name}")


def infer_wild_type(sequences: List[str]) -> str:
    if len(sequences) == 0:
        raise ValueError("Empty sequence list")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError("Sequences have inconsistent lengths; expected fixed-length variants")

    seq_len = len(sequences[0])
    wt_chars = []
    for pos in range(seq_len):
        counts = {}
        for s in sequences:
            ch = s[pos]
            counts[ch] = counts.get(ch, 0) + 1
        wt_chars.append(max(counts.items(), key=lambda x: x[1])[0])
    return "".join(wt_chars)


def build_targets(sequences: List[str], wt_seq: str):
    """
    mut_mask: [N, L] float32, 1 if mutated else 0
    aa_target: [N, L] int64, amino-acid class index for mutated positions,
               -100 for unchanged positions
    """
    n = len(sequences)
    L = len(wt_seq)
    mut_mask = np.zeros((n, L), dtype=np.float32)
    aa_target = np.full((n, L), fill_value=-100, dtype=np.int64)

    for i, seq in enumerate(sequences):
        if len(seq) != L:
            raise ValueError("Sequence length mismatch vs wild-type")
        for j, (wt_ch, ch) in enumerate(zip(wt_seq, seq)):
            if ch != wt_ch:
                mut_mask[i, j] = 1.0
                if ch not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid '{ch}' in sequence {i}")
                aa_target[i, j] = AA_TO_IDX[ch]

    return mut_mask, aa_target


class MutationDecoder(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, hidden_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mask_head = nn.Linear(hidden_dim, seq_len)          # [B, L]
        self.aa_head = nn.Linear(hidden_dim, seq_len * 20)       # [B, L*20]

    def forward(self, z: torch.Tensor):
        h = self.shared(z)
        mask_logits = self.mask_head(h)
        aa_logits = self.aa_head(h).view(-1, self.seq_len, 20)
        return mask_logits, aa_logits


@dataclass
class Config:
    dense_npz: str
    latent_model_ckpt: str
    latent_model_name: str
    latent_dim: int
    latent_hidden_dim: int
    decoder_hidden_dim: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    mutation_loss_weight: float
    aa_loss_weight: float
    mask_pos_weight_scale: float
    seed: int
    test_size: float
    val_size: float
    device: str
    output_prefix: str


def create_batches(n: int, batch_size: int, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def compute_sequence_metrics(mask_true, aa_true, mask_logits, aa_logits):
    mask_prob = torch.sigmoid(mask_logits)
    mask_pred = (mask_prob > 0.5).float()

    mask_true_np = mask_true.cpu().numpy()
    mask_pred_np = mask_pred.cpu().numpy()

    tp = float(((mask_true_np == 1) & (mask_pred_np == 1)).sum())
    fp = float(((mask_true_np == 0) & (mask_pred_np == 1)).sum())
    fn = float(((mask_true_np == 1) & (mask_pred_np == 0)).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    aa_pred = aa_logits.argmax(dim=-1)
    valid_mut_positions = (aa_true != -100)
    if valid_mut_positions.sum().item() > 0:
        mut_aa_acc = (
            (aa_pred[valid_mut_positions] == aa_true[valid_mut_positions]).float().mean().item()
        )
    else:
        mut_aa_acc = float("nan")

    pos_acc = ((mask_pred == mask_true).float()).mean().item()

    return {
        "mutation_precision": precision,
        "mutation_recall": recall,
        "mutation_f1": f1,
        "mutated_aa_accuracy": mut_aa_acc,
        "position_accuracy": pos_acc,
    }


def encode_latents(encoder, X: np.ndarray, batch_size: int, device: str):
    encoder.eval()
    out = []
    with torch.no_grad():
        for idx in create_batches(len(X), batch_size=batch_size, shuffle=False):
            xb = torch.from_numpy(X[idx]).to(device)
            z = encoder.encode_continuous(xb)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_mask_pos_weight(mut_mask_train: np.ndarray, scale: float) -> float:
    pos = float(mut_mask_train.sum())
    total = float(mut_mask_train.size)
    neg = total - pos
    if pos <= 0:
        return 1.0
    return max((neg / pos) * scale, 1.0)


def evaluate_decoder(
    model,
    Z,
    mut_mask,
    aa_target,
    batch_size,
    device,
    mutation_loss_weight,
    aa_loss_weight,
    mask_pos_weight
):
    model.eval()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_aa_loss = 0.0
    n = 0

    pos_weight_tensor = torch.tensor(mask_pos_weight, dtype=torch.float32, device=device)

    all_mask_true = []
    all_aa_true = []
    all_mask_logits = []
    all_aa_logits = []

    with torch.no_grad():
        for idx in create_batches(len(Z), batch_size=batch_size, shuffle=False):
            zb = torch.from_numpy(Z[idx]).to(device)
            mask_true = torch.from_numpy(mut_mask[idx]).to(device)
            aa_true = torch.from_numpy(aa_target[idx]).to(device)

            mask_logits, aa_logits = model(zb)

            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_true,
                pos_weight=pos_weight_tensor
            )

            aa_loss = F.cross_entropy(
                aa_logits.reshape(-1, 20),
                aa_true.reshape(-1),
                ignore_index=-100,
            )

            loss = mutation_loss_weight * mask_loss + aa_loss_weight * aa_loss

            bsz = zb.shape[0]
            total_loss += float(loss.item()) * bsz
            total_mask_loss += float(mask_loss.item()) * bsz
            total_aa_loss += float(aa_loss.item()) * bsz
            n += bsz

            all_mask_true.append(mask_true.cpu())
            all_aa_true.append(aa_true.cpu())
            all_mask_logits.append(mask_logits.cpu())
            all_aa_logits.append(aa_logits.cpu())

    all_mask_true = torch.cat(all_mask_true, dim=0)
    all_aa_true = torch.cat(all_aa_true, dim=0)
    all_mask_logits = torch.cat(all_mask_logits, dim=0)
    all_aa_logits = torch.cat(all_aa_logits, dim=0)

    metrics = compute_sequence_metrics(
        all_mask_true, all_aa_true, all_mask_logits, all_aa_logits
    )
    metrics.update({
        "loss": total_loss / max(n, 1),
        "mask_loss": total_mask_loss / max(n, 1),
        "aa_loss": total_aa_loss / max(n, 1),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-npz", type=str, required=True)
    parser.add_argument("--latent-model-ckpt", type=str, required=True)
    parser.add_argument("--latent-model-name", type=str, choices=["ae", "vae"], required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--latent-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mutation-loss-weight", type=float, default=2.0)
    parser.add_argument("--aa-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-prefix", type=str, required=True)
    args = parser.parse_args()

    config = Config(
        dense_npz=args.dense_npz,
        latent_model_ckpt=args.latent_model_ckpt,
        latent_model_name=args.latent_model_name,
        latent_dim=args.latent_dim,
        latent_hidden_dim=args.latent_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mutation_loss_weight=args.mutation_loss_weight,
        aa_loss_weight=args.aa_loss_weight,
        mask_pos_weight_scale=args.mask_pos_weight_scale,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        device=args.device,
        output_prefix=args.output_prefix,
    )

    set_seed(config.seed)

    X, y, items = load_dense_npz(config.dense_npz)
    sequences = [str(s) for s in items.tolist()]
    wt_seq = infer_wild_type(sequences)
    seq_len = len(wt_seq)

    X_trainval, X_test, seq_trainval, seq_test = train_test_split(
        X, sequences, test_size=config.test_size, random_state=config.seed
    )
    val_ratio_adjusted = config.val_size / (1.0 - config.test_size)
    X_train, X_val, seq_train, seq_val = train_test_split(
        X_trainval, seq_trainval, test_size=val_ratio_adjusted, random_state=config.seed
    )

    mut_train, aa_train = build_targets(seq_train, wt_seq)
    mut_val, aa_val = build_targets(seq_val, wt_seq)
    mut_test, aa_test = build_targets(seq_test, wt_seq)

    train_mutation_rate = float(mut_train.mean())
    mask_pos_weight = compute_mask_pos_weight(mut_train, config.mask_pos_weight_scale)

    latent_ckpt = load_latent_checkpoint(config.latent_model_ckpt, config.device)
    input_dim = latent_ckpt["input_dim"]

    encoder = build_latent_encoder(
        config.latent_model_name,
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.latent_hidden_dim,
    ).to(config.device)
    encoder.load_state_dict(latent_ckpt["state_dict"])
    encoder.eval()

    Z_train = encode_latents(encoder, X_train, config.batch_size, config.device)
    Z_val = encode_latents(encoder, X_val, config.batch_size, config.device)
    Z_test = encode_latents(encoder, X_test, config.batch_size, config.device)

    decoder = MutationDecoder(
        latent_dim=config.latent_dim,
        seq_len=seq_len,
        hidden_dim=config.decoder_hidden_dim,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        decoder.train()
        pos_weight_tensor = torch.tensor(mask_pos_weight, dtype=torch.float32, device=config.device)

        for idx in create_batches(len(Z_train), config.batch_size, shuffle=True):
            zb = torch.from_numpy(Z_train[idx]).to(config.device)
            mask_true = torch.from_numpy(mut_train[idx]).to(config.device)
            aa_true = torch.from_numpy(aa_train[idx]).to(config.device)

            mask_logits, aa_logits = decoder(zb)

            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_true,
                pos_weight=pos_weight_tensor
            )
            aa_loss = F.cross_entropy(
                aa_logits.reshape(-1, 20),
                aa_true.reshape(-1),
                ignore_index=-100,
            )
            loss = config.mutation_loss_weight * mask_loss + config.aa_loss_weight * aa_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_decoder(
            decoder, Z_train, mut_train, aa_train, config.batch_size, config.device,
            config.mutation_loss_weight, config.aa_loss_weight, mask_pos_weight
        )
        val_metrics = evaluate_decoder(
            decoder, Z_val, mut_val, aa_val, config.batch_size, config.device,
            config.mutation_loss_weight, config.aa_loss_weight, mask_pos_weight
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mutation_f1": train_metrics["mutation_f1"],
            "train_mutated_aa_accuracy": train_metrics["mutated_aa_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_mutation_f1": val_metrics["mutation_f1"],
            "val_mutated_aa_accuracy": val_metrics["mutated_aa_accuracy"],
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"[decoder] epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mut_f1={val_metrics['mutation_f1']:.4f} "
                f"val_mut_aa_acc={val_metrics['mutated_aa_accuracy']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("No best decoder state found")
    decoder.load_state_dict(best_state)

    train_metrics = evaluate_decoder(
        decoder, Z_train, mut_train, aa_train, config.batch_size, config.device,
        config.mutation_loss_weight, config.aa_loss_weight, mask_pos_weight
    )
    val_metrics = evaluate_decoder(
        decoder, Z_val, mut_val, aa_val, config.batch_size, config.device,
        config.mutation_loss_weight, config.aa_loss_weight, mask_pos_weight
    )
    test_metrics = evaluate_decoder(
        decoder, Z_test, mut_test, aa_test, config.batch_size, config.device,
        config.mutation_loss_weight, config.aa_loss_weight, mask_pos_weight
    )

    os.makedirs(os.path.dirname(config.output_prefix), exist_ok=True)

    model_path = config.output_prefix + ".pt"
    metrics_path = config.output_prefix + ".json"

    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "config": asdict(config),
            "wt_seq": wt_seq,
            "seq_len": seq_len,
            "mask_pos_weight": mask_pos_weight,
        },
        model_path,
    )

    metrics = {
        "config": asdict(config),
        "wt_seq": wt_seq,
        "seq_len": seq_len,
        "n_train": len(Z_train),
        "n_val": len(Z_val),
        "n_test": len(Z_test),
        "train_mutation_rate": train_mutation_rate,
        "mask_pos_weight": mask_pos_weight,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved decoder model to: {model_path}")
    print(f"Saved decoder metrics to: {metrics_path}")
    print(json.dumps({
        "train_mutation_rate": train_mutation_rate,
        "mask_pos_weight": mask_pos_weight,
        "test": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
