import argparse
import os
import pandas as pd


def fmt(x, digits=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-csv", required=True)
    parser.add_argument("--decoder-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    latent_df = pd.read_csv(args.latent_csv)
    decoder_df = pd.read_csv(args.decoder_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================
    # Table 3a: AE / VAE collapse
    # ============================
    latent_df["representation"] = latent_df["representation"].str.upper()
    latent_df = latent_df[latent_df["representation"].isin(["AE", "VAE"])]

    latent_df = latent_df.sort_values(
        ["dataset", "train_size", "representation", "latent_dim"]
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Latent quality of learned representations. "
        r"Although AE and VAE achieve low reconstruction error, their binarized latent codes exhibit near-zero entropy and inactive dimensions, indicating collapse.}"
    )
    lines.append(r"\label{tab:latent_collapse}")
    lines.append(r"\begin{tabular}{l l l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Train size & Representation & Latent dim & Bit entropy $\uparrow$ & Active dims $\uparrow$ & Recon. MSE $\downarrow$ \\")
    lines.append(r"\midrule")

    current = None
    for _, row in latent_df.iterrows():
        key = (row["dataset"], row["train_size"])
        if current is not None and key != current:
            lines.append(r"\midrule")
        current = key

        lines.append(
            f"{row['dataset'].upper()} & {int(row['train_size'])} & {row['representation']} & {int(row['latent_dim'])} & "
            f"{fmt(row['test_bit_entropy'])} & {fmt(row['test_active_dims'],2)} & {fmt(row['test_recon_mse'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(args.output_dir, "table3a_latent.tex"), "w") as f:
        f.write("\n".join(lines))

    print("Saved Table 3a")

    # ============================
    # Table 3b: Decoder quality
    # ============================
    decoder_df["representation"] = decoder_df["latent_type"].map({
        "pca": "PCA",
        "random": "Random"
    })

    # pick best optimizer per setting
    decoder_df = decoder_df.sort_values(
        ["dataset", "train_size", "representation", "latent_dim", "f1"],
        ascending=[True, True, True, True, False]
    )

    decoder_df = decoder_df.groupby(
        ["dataset", "train_size", "representation", "latent_dim"],
        as_index=False
    ).first()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Decoding performance of binary latent representations. "
        r"PCA-based representations achieve substantially higher mutation F1 and amino-acid accuracy than random projections, "
        r"and improve with increasing latent dimensionality.}"
    )
    lines.append(r"\label{tab:decoder_quality}")
    lines.append(r"\begin{tabular}{l l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Train size & Representation & Latent dim & Mutation F1 $\uparrow$ & AA accuracy $\uparrow$ \\")
    lines.append(r"\midrule")

    current = None
    for _, row in decoder_df.iterrows():
        key = (row["dataset"], row["train_size"])
        if current is not None and key != current:
            lines.append(r"\midrule")
        current = key

        lines.append(
            f"{row['dataset'].upper()} & {int(row['train_size'])} & {row['representation']} & {int(row['latent_dim'])} & "
            f"{fmt(row['f1'])} & {fmt(row['aa_acc'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(args.output_dir, "table3b_decoder.tex"), "w") as f:
        f.write("\n".join(lines))

    print("Saved Table 3b")


if __name__ == "__main__":
    main()
