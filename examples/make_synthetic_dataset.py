import argparse
from src.data.synthetic import save_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="examples/synthetic_peptides.npz")
    parser.add_argument("--N", type=int, default=300)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--peptide-length", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--interaction-scale", type=float, default=0.2)
    parser.add_argument("--sparse-interactions", action="store_true")
    parser.add_argument("--interaction-keep-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    save_dataset(
        args.out,
        N=args.N,
        M=args.M,
        peptide_length=args.peptide_length,
        noise_std=args.noise_std,
        interaction_scale=args.interaction_scale,
        sparse_interactions=args.sparse_interactions,
        interaction_keep_prob=args.interaction_keep_prob,
        seed=args.seed,
    )
    print(f"Saved synthetic dataset to {args.out}")


if __name__ == "__main__":
    main()
