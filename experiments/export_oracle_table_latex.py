import argparse
import pandas as pd


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="artifacts/results/oracle_grouped_summary.csv",
    )
    parser.add_argument(
        "--output-tex",
        default="artifacts/results/oracle_table_rows.tex",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv).copy()

    rows = []
    for _, r in df.iterrows():
        row = (
            f"{r['dataset'].upper()} & "
            f"{int(r['train_size'])} & "
            f"{r['model']} & "
            f"{fmt(r['spearman'])} & "
            f"{fmt(r['pearson'])} & "
            f"{fmt(r['rmse'])} & "
            f"{fmt(r['mae'])} \\\\"
        )
        rows.append(row)

    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Saved LaTeX rows to: {args.output_tex}")


if __name__ == "__main__":
    main()
