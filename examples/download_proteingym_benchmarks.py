import argparse
import io
import os
import re
import zipfile
from typing import Dict, List, Optional

import pandas as pd
import requests


INDEX_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv"
ZIP_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"

DEFAULT_BENCHMARKS = {
    "gfp": ["gfp", "sarkisyan"],
    "gb1": ["olson"],
    "aav": ["aav", "matreyek"],
}


def normalize_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return x.strip()


def build_search_text(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(normalize_text(row[c]))
    return " ".join(parts)


def choose_row(index_df: pd.DataFrame, keywords: List[str]) -> Optional[pd.Series]:
    keywords = [normalize_text(k) for k in keywords]
    candidate_cols = [
        c for c in [
            "DMS_id",
            "DMS_filename",
            "UniProt_ID",
            "molecule_name",
            "target_seq",
            "first_author",
            "title",
        ]
        if c in index_df.columns
    ]

    matches = []
    for _, row in index_df.iterrows():
        text = build_search_text(row, candidate_cols)
        if all(k in text for k in keywords):
            matches.append(row)

    if not matches:
        return None

    # Prefer assays with more mutants if available
    if "DMS_total_number_mutants" in index_df.columns:
        matches = sorted(
            matches,
            key=lambda r: float(r.get("DMS_total_number_mutants", 0) or 0),
            reverse=True,
        )

    return matches[0]


def download_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def read_csv_from_zip(zip_bytes: bytes, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()

        # exact match first
        if filename in names:
            with zf.open(filename) as f:
                return pd.read_csv(f)

        # fallback: basename match
        base = os.path.basename(filename)
        for name in names:
            if os.path.basename(name) == base:
                with zf.open(name) as f:
                    return pd.read_csv(f)

        raise FileNotFoundError(
            f"Could not find '{filename}' inside ZIP archive. "
            f"Example names in archive: {names[:10]}"
        )


def convert_assay_df(df: pd.DataFrame) -> pd.DataFrame:
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
    out = out.dropna(subset=["sequence", "fitness"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data/proteingym")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["gfp", "gb1", "aav"],
        help="Subset of benchmarks to download: gfp gb1 aav",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only resolve and print matching ProteinGym rows.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Downloading index: {INDEX_URL}")
    index_df = pd.read_csv(INDEX_URL)
    print(f"Loaded {len(index_df)} assay rows")
    print("Index columns:", list(index_df.columns))
    print()

    resolved: Dict[str, pd.Series] = {}

    for name in args.benchmarks:
        key = name.lower()
        if key not in DEFAULT_BENCHMARKS:
            raise ValueError(f"Unknown benchmark '{name}'. Choose from {list(DEFAULT_BENCHMARKS)}")

        row = choose_row(index_df, DEFAULT_BENCHMARKS[key])
        if row is None:
            print(f"[WARN] Could not resolve benchmark '{key}'")
            continue

        resolved[key] = row
        print(f"{key} -> DMS_id={row['DMS_id']} | DMS_filename={row['DMS_filename']}")

    if args.print_only:
        return

    print("\nDownloading ZIP archive once...")
    zip_bytes = download_bytes(ZIP_URL)

    for key, row in resolved.items():
        filename = str(row["DMS_filename"])
        print(f"\nExtracting {key}: {filename}")

        assay_df = read_csv_from_zip(zip_bytes, filename)
        out_df = convert_assay_df(assay_df)

        output_csv = os.path.join(args.outdir, f"{key}.csv")
        out_df.to_csv(output_csv, index=False)

        print(f"Saved {len(out_df)} rows to {output_csv}")
        print(out_df.head())


if __name__ == "__main__":
    main()
