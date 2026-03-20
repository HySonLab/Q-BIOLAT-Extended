import argparse
import json
import os
import subprocess
import sys


def run_command(cmd):
    print(f"\n[RUN] {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="examples/synthetic_peptides.npz")
    parser.add_argument("--outdir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    run_command([sys.executable, "experiments/train_surrogate.py", "--data", args.data, "--model", "qubo", "--out", os.path.join(args.outdir, "qubo_metrics.json")])
    run_command([sys.executable, "experiments/train_surrogate.py", "--data", args.data, "--model", "mlp", "--out", os.path.join(args.outdir, "mlp_metrics.json")])
    run_command([sys.executable, "experiments/optimize_latent.py", "--data", args.data, "--out", os.path.join(args.outdir, "optimization_results.json")])

    summary = {
        "status": "completed",
        "artifacts_dir": args.outdir,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
