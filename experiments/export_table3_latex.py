import argparse
import os
import pandas as pd


def latex_escape(s):
    return str(s).replace("_", r"\_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-csv", required=True)
    parser.add_argument("--latent-csv", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    decoder_df = pd.read_csv(args.decoder_csv)
    latent_df = pd.read_csv(args.latent_csv)

    # Merge on common keys
    df = pd.merge(
        decoder_df,
        latent_df,
        on=["dataset", "train_size", "representation", "latent_dim"],
        how="inner",
    )

    # Rename columns for clarity
    df = df.rename(columns={
        "representation": "rep",
        "test_bit_entropy": "bit_entropy",
        "test_active_dims": "active_dims",
        "test_recon_mse": "recon_mse",
        "f1": "mutation_f1",
        "aa_acc": "aa_acc",
    })

    # Sort nicely
    df = df.sort_values(
        ["dataset", "train_size", "rep", "latent_dim"]
    )

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of latent representation methods and decoding performance. "
                 r"We report latent quality metrics (bit entropy and number of active latent dimensions) "
                 r"and decoding metrics (mutation F1 and mutated-residue accuracy).}")
    lines.append(r"\label{tab:latent_decoder}")
    lines.append(r"\begin{tabular}{l l l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Train size & Representation & Latent dim & Bit entropy $\uparrow$ & Active dims $\uparrow$ & Mutation F1 $\uparrow$ & AA acc $\uparrow$ \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        dataset = row["dataset"].upper()
        train_size = int(row["train_size"])
        rep = latex_escape(row["rep"])
        dim = int(row["latent_dim"])

        bit_entropy = f"{row['bit_entropy']:.3f}"
        active_dims = f"{row['active_dims']:.2f}"
        f1 = f"{row['mutation_f1']:.3f}"
        aa = f"{row['aa_acc']:.3f}"

        lines.append(
            f"{dataset} & {train_size} & {rep} & {dim} & {bit_entropy} & {active_dims} & {f1} & {aa} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w") as f:
        f.write("\n".join(lines))

    print("Saved Table 3 LaTeX to:", args.output_tex)


if __name__ == "__main__":
    main()
