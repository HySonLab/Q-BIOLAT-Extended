import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.n > len(df):
        raise ValueError(f"Requested n={args.n}, but dataset only has {len(df)} rows.")

    subset = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    subset.to_csv(args.output_csv, index=False)

    print(f"Input rows : {len(df)}")
    print(f"Output rows: {len(subset)}")
    print(f"Saved to   : {args.output_csv}")


if __name__ == "__main__":
    main()
