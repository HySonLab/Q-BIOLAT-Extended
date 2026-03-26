import argparse
import os
import pandas as pd


REP_MAP = {
    "AE": "AE",
    "VAE": "VAE",
    "pca": "PCA",
    "random": "Random projection",
}


def normalize_rep(x: str) -> str:
    x = str(x)
    return REP_MAP.get(x, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-csv",
        default="artifacts/results/table3_latent_summary.csv",
        help="CSV with AE/VAE latent metrics",
    )
    parser.add_argument(
        "--decoder-csv",
        default="artifacts/results/decoder_summary.csv",
        help="CSV with PCA/Random decoder metrics",
    )
    parser.add_argument(
        "--output-tex",
        default="artifacts/results/table3.tex",
        help="Output LaTeX file",
    )
    args = parser.parse_args()

    latent_df = pd.read_csv(args.latent_csv)
    decoder_df = pd.read_csv(args.decoder_csv)

    # -----------------------------
    # AE / VAE rows from latent CSV
    # -----------------------------
    latent_rows = latent_df.copy()
    latent_rows["representation"] = latent_rows["representation"].map(normalize_rep)

    latent_rows = latent_rows[
        ["dataset", "train_size", "representation", "latent_dim",
         "test_bit_entropy", "test_active_dims", "test_recon_mse"]
    ].copy()

    latent_rows["mutation_f1"] = pd.NA
    latent_rows["aa_acc"] = pd.NA

    # ---------------------------------
    # PCA / Random rows from decoder CSV
    # ---------------------------------
    decoder_rows = decoder_df.copy()
    decoder_rows["representation"] = decoder_rows["latent_type"].map(normalize_rep)

    decoder_rows = decoder_rows[
        ["dataset", "train_size", "representation", "latent_dim",
         "f1", "aa_acc"]
    ].copy()

    decoder_rows = decoder_rows.rename(columns={"f1": "mutation_f1"})

    decoder_rows["test_bit_entropy"] = pd.NA
    decoder_rows["test_active_dims"] = pd.NA
    decoder_rows["test_recon_mse"] = pd.NA

    # ---------------------------------
    # Merge into one display table
    # ---------------------------------
    df = pd.concat([latent_rows, decoder_rows], ignore_index=True)

    # keep ordering nice
    rep_order = {
        "Random projection": 0,
        "PCA": 1,
        "AE": 2,
        "VAE": 3,
    }
    df["rep_order"] = df["representation"].map(rep_order).fillna(999)

    df = df.sort_values(
        ["dataset", "train_size", "rep_order", "latent_dim"]
    ).drop(columns=["rep_order"])

    def fmt(x, digits=3):
        if pd.isna(x):
            return "--"
        return f"{x:.{digits}f}"

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Comparison of latent representation methods and decoding performance. "
        r"We report latent quality metrics (bit entropy and number of active latent dimensions), "
        r"reconstruction error for learned latent models, and decoding quality metrics "
        r"(mutation F1 and mutated-residue accuracy). AE and VAE representations collapse after "
        r"binarization, while PCA supports substantially stronger decoding performance than random projection.}"
    )
    lines.append(r"\label{tab:latent_decoder}")
    lines.append(r"\begin{tabular}{l l l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Train size & Representation & Latent dim & "
        r"Bit entropy $\uparrow$ & Active dims $\uparrow$ & Recon. MSE $\downarrow$ & "
        r"Mutation F1 $\uparrow$ & AA accuracy $\uparrow$ \\"
    )
    lines.append(r"\midrule")

    current_dataset = None
    current_train = None

    for _, row in df.iterrows():
        dataset = str(row["dataset"]).upper()
        train_size = int(row["train_size"])
        representation = row["representation"]
        latent_dim = int(row["latent_dim"])

        if current_dataset is not None and (
            dataset != current_dataset or train_size != current_train
        ):
            lines.append(r"\midrule")

        current_dataset = dataset
        current_train = train_size

        lines.append(
            f"{dataset} & {train_size} & {representation} & {latent_dim} & "
            f"{fmt(row['test_bit_entropy'])} & "
            f"{fmt(row['test_active_dims'], digits=2)} & "
            f"{fmt(row['test_recon_mse'])} & "
            f"{fmt(row['mutation_f1'])} & "
            f"{fmt(row['aa_acc'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved Table 3 LaTeX to: {args.output_tex}")


if __name__ == "__main__":
    main()
