import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


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

    def encode_binary(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_continuous(x)
        return (z > 0).float()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode_continuous(x)
        xhat = self.decoder(z)
        return {"z": z, "xhat": xhat}


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

    def encode_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu

    def encode_binary(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_continuous(x)
        return (z > 0).float()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return {"mu": mu, "logvar": logvar, "z": z, "xhat": xhat}


@dataclass
class Config:
    input_npz: str
    model_name: str
    latent_dim: int
    hidden_dim: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    beta_kl: float
    seed: int
    test_size: float
    val_size: float
    device: str
    output_prefix: str


def make_model(model_name: str, input_dim: int, latent_dim: int, hidden_dim: int) -> nn.Module:
    if model_name == "ae":
        return AENet(input_dim, latent_dim, hidden_dim)
    if model_name == "vae":
        return VAENet(input_dim, latent_dim, hidden_dim)
    raise ValueError(f"Unknown model_name: {model_name}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def create_batches(X: np.ndarray, batch_size: int, shuffle: bool = True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx]


def compute_binary_stats(B: np.ndarray) -> Dict[str, object]:
    """
    B: binary latent matrix of shape [n_samples, latent_dim]
    """
    bit_prob = B.mean(axis=0)  # probability each bit is 1 over samples
    frac_ones = float(B.mean())

    # entropy per dimension; max is ln(2) ~ 0.693 when p = 0.5
    eps = 1e-8
    entropy = -(
        bit_prob * np.log(bit_prob + eps) +
        (1.0 - bit_prob) * np.log(1.0 - bit_prob + eps)
    )
    bit_entropy = float(np.mean(entropy))

    # dimensions that are actually active across samples
    active_dims = float(np.mean((bit_prob > 0.05) & (bit_prob < 0.95)))

    # optional: exact count as well
    active_dim_count = int(np.sum((bit_prob > 0.05) & (bit_prob < 0.95)))

    return {
        "bit_entropy": bit_entropy,
        "active_dims": active_dims,
        "active_dim_count": active_dim_count,
        "frac_ones": frac_ones,
        "per_dim_one_rate": bit_prob.tolist(),
    }


def evaluate_model(model: nn.Module, X: np.ndarray, batch_size: int, model_name: str, beta_kl: float, device: str):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n = 0

    latents = []
    binaries = []

    with torch.no_grad():
        for xb_np in create_batches(X, batch_size=batch_size, shuffle=False):
            xb = torch.from_numpy(xb_np).to(device)
            out = model(xb)
            recon = F.mse_loss(out["xhat"], xb, reduction="mean")

            if model_name == "vae":
                kl = kl_divergence(out["mu"], out["logvar"])
                loss = recon + beta_kl * kl
                z_cont = out["mu"]
            else:
                kl = torch.tensor(0.0, device=device)
                loss = recon
                z_cont = out["z"]

            z_bin = (z_cont > 0).float()

            bsz = xb.shape[0]
            total_loss += float(loss.item()) * bsz
            total_recon += float(recon.item()) * bsz
            total_kl += float(kl.item()) * bsz
            n += bsz

            latents.append(z_cont.cpu().numpy())
            binaries.append(z_bin.cpu().numpy())

    Z = np.concatenate(latents, axis=0)
    B = np.concatenate(binaries, axis=0)

    binary_stats = compute_binary_stats(B)

    return {
        "loss": total_loss / max(n, 1),
        "recon_mse": total_recon / max(n, 1),
        "kl": total_kl / max(n, 1),
        "bit_entropy": binary_stats["bit_entropy"],
        "active_dims": binary_stats["active_dims"],
        "active_dim_count": binary_stats["active_dim_count"],
        "frac_ones": binary_stats["frac_ones"],
        "per_dim_one_rate": binary_stats["per_dim_one_rate"],
        "Z": Z,
        "B": B,
    }


def train(config: Config):
    set_seed(config.seed)

    X, y, items = load_dense_npz(config.input_npz)
    input_dim = X.shape[1]

    X_trainval, X_test, y_trainval, y_test, items_trainval, items_test = train_test_split(
        X, y, items, test_size=config.test_size, random_state=config.seed
    )
    val_ratio_adjusted = config.val_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val, items_train, items_val = train_test_split(
        X_trainval, y_trainval, items_trainval, test_size=val_ratio_adjusted, random_state=config.seed
    )

    model = make_model(config.model_name, input_dim, config.latent_dim, config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()

        for xb_np in create_batches(X_train, batch_size=config.batch_size, shuffle=True):
            xb = torch.from_numpy(xb_np).to(config.device)

            out = model(xb)
            recon = F.mse_loss(out["xhat"], xb, reduction="mean")

            if config.model_name == "vae":
                kl = kl_divergence(out["mu"], out["logvar"])
                loss = recon + config.beta_kl * kl
            else:
                kl = torch.tensor(0.0, device=config.device)
                loss = recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(
            model, X_train, config.batch_size, config.model_name, config.beta_kl, config.device
        )
        val_metrics = evaluate_model(
            model, X_val, config.batch_size, config.model_name, config.beta_kl, config.device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_recon_mse": train_metrics["recon_mse"],
            "train_kl": train_metrics["kl"],
            "train_bit_entropy": train_metrics["bit_entropy"],
            "train_active_dims": train_metrics["active_dims"],
            "train_frac_ones": train_metrics["frac_ones"],
            "val_loss": val_metrics["loss"],
            "val_recon_mse": val_metrics["recon_mse"],
            "val_kl": val_metrics["kl"],
            "val_bit_entropy": val_metrics["bit_entropy"],
            "val_active_dims": val_metrics["active_dims"],
            "val_frac_ones": val_metrics["frac_ones"],
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"[{config.model_name}] epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_recon={val_metrics['recon_mse']:.6f} "
                f"val_entropy={val_metrics['bit_entropy']:.4f} "
                f"val_active_dims={val_metrics['active_dims']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training failed: no best model state found.")

    model.load_state_dict(best_state)

    train_metrics = evaluate_model(
        model, X_train, config.batch_size, config.model_name, config.beta_kl, config.device
    )
    val_metrics = evaluate_model(
        model, X_val, config.batch_size, config.model_name, config.beta_kl, config.device
    )
    test_metrics = evaluate_model(
        model, X_test, config.batch_size, config.model_name, config.beta_kl, config.device
    )

    os.makedirs(os.path.dirname(config.output_prefix), exist_ok=True)

    model_path = config.output_prefix + ".pt"
    metrics_path = config.output_prefix + ".json"
    latent_path = config.output_prefix + "_test_latents.npz"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "input_dim": input_dim,
        },
        model_path,
    )

    metrics = {
        "config": asdict(config),
        "input_dim": input_dim,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train": {
            "loss": train_metrics["loss"],
            "recon_mse": train_metrics["recon_mse"],
            "kl": train_metrics["kl"],
            "bit_entropy": train_metrics["bit_entropy"],
            "active_dims": train_metrics["active_dims"],
            "active_dim_count": train_metrics["active_dim_count"],
            "frac_ones": train_metrics["frac_ones"],
            "per_dim_one_rate": train_metrics["per_dim_one_rate"],
        },
        "val": {
            "loss": val_metrics["loss"],
            "recon_mse": val_metrics["recon_mse"],
            "kl": val_metrics["kl"],
            "bit_entropy": val_metrics["bit_entropy"],
            "active_dims": val_metrics["active_dims"],
            "active_dim_count": val_metrics["active_dim_count"],
            "frac_ones": val_metrics["frac_ones"],
            "per_dim_one_rate": val_metrics["per_dim_one_rate"],
        },
        "test": {
            "loss": test_metrics["loss"],
            "recon_mse": test_metrics["recon_mse"],
            "kl": test_metrics["kl"],
            "bit_entropy": test_metrics["bit_entropy"],
            "active_dims": test_metrics["active_dims"],
            "active_dim_count": test_metrics["active_dim_count"],
            "frac_ones": test_metrics["frac_ones"],
            "per_dim_one_rate": test_metrics["per_dim_one_rate"],
        },
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        latent_path,
        z_cont=test_metrics["Z"].astype(np.float32),
        z_bin=test_metrics["B"].astype(np.float32),
        y=y_test.astype(np.float32),
        items=items_test,
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved test latents to: {latent_path}")
    print(json.dumps({
        "test_recon_mse": test_metrics["recon_mse"],
        "test_bit_entropy": test_metrics["bit_entropy"],
        "test_active_dims": test_metrics["active_dims"],
        "test_frac_ones": test_metrics["frac_ones"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--model-name", type=str, choices=["ae", "vae"], required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-prefix", type=str, required=True)
    args = parser.parse_args()

    config = Config(
        input_npz=args.input_npz,
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_kl=args.beta_kl,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        device=args.device,
        output_prefix=args.output_prefix,
    )

    train(config)


if __name__ == "__main__":
    main()
