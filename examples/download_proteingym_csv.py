# examples/download_proteingym_csv.py
import argparse
import os
import pandas as pd


DEFAULT_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-url",
        type=str,
        default=DEFAULT_URL,
        help="Direct URL to a ProteinGym substitutions CSV.",
    )
    parser.add_argument(
        "--dms-id",
        type=str,
        default=None,
        help="Optional exact DMS_id to filter, e.g. GFP_AEQVI_Sarkisyan_2016.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to keep after filtering.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save CSV with sequence and fitness.",
    )
    parser.add_argument(
        "--include-extra-cols",
        action="store_true",
        help="Keep extra ProteinGym columns in the output.",
    )
    args = parser.parse_args()

    print(f"Downloading ProteinGym table from: {args.input_url}")
    df = pd.read_csv(args.input_url)
    print(f"Loaded {len(df)} rows")
    print("Columns:", list(df.columns))

    if "DMS_id" in df.columns:
        print("\nTop available DMS_id values:")
        print(df["DMS_id"].value_counts().head(20))

    if args.dms_id is not None:
        if "DMS_id" not in df.columns:
            raise ValueError("Column DMS_id is missing; cannot filter by --dms-id.")
        df = df[df["DMS_id"] == args.dms_id].copy()
        print(f"\nRows after filtering DMS_id={args.dms_id}: {len(df)}")
        if len(df) == 0:
            raise ValueError(
                f"No rows found for DMS_id={args.dms_id}. "
                "Run once without --dms-id to inspect valid IDs."
            )

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Keeping first {len(df)} rows due to --limit")

    required = ["mutated_sequence", "DMS_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out = pd.DataFrame(
        {
            "sequence": df["mutated_sequence"].astype(str),
            "fitness": pd.to_numeric(df["DMS_score"], errors="coerce"),
        }
    )

    if args.include_extra_cols:
        extra_cols = [c for c in ["DMS_id", "mutant", "mutated_sequence", "DMS_score_bin"] if c in df.columns]
        for col in extra_cols:
            out[col] = df[col]

    out = out.dropna(subset=["sequence", "fitness"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"\nSaved {len(out)} rows to {args.output_csv}")
    print("\nPreview:")
    print(out.head())


if __name__ == "__main__":
    main()
