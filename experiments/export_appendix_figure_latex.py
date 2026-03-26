import argparse
import os


def make_figure_block(dataset: str, train_sizes, plot_dir: str, label: str, caption: str) -> str:
    lines = []
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")

    for i, train_size in enumerate(train_sizes):
        filename = f"{dataset}_{train_size}_methods_plot.png"
        path = os.path.join(plot_dir, filename).replace("\\", "/")

        lines.append(r"\begin{subfigure}[t]{0.48\textwidth}")
        lines.append(r"\centering")
        lines.append(rf"\includegraphics[width=\linewidth]{{{path}}}")
        lines.append(rf"\caption{{{dataset.upper()} with {train_size} training samples.}}")
        lines.append(r"\end{subfigure}")

        if i % 2 == 0:
            lines.append(r"\hfill")
        else:
            lines.append(r"\vspace{0.5em}")

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        required=True,
        help="Directory containing the PNG plots",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save LaTeX figure blocks",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gfp_caption = (
        "End-to-end design performance on GFP across different training sizes. "
        "The x-axis shows the number of bits and the y-axis shows the best oracle score. "
        "Each line corresponds to an optimizer-representation pair, including simulated annealing (SA), "
        "genetic algorithm (GA), random search (RS), and greedy hill climbing (GHC), combined with either "
        "PCA-based or random-projection binary latent representations."
    )

    aav_caption = (
        "End-to-end design performance on AAV across different training sizes. "
        "The x-axis shows the number of bits and the y-axis shows the best oracle score. "
        "Each line corresponds to an optimizer-representation pair, including simulated annealing (SA), "
        "genetic algorithm (GA), random search (RS), and greedy hill climbing (GHC), combined with either "
        "PCA-based or random-projection binary latent representations."
    )

    gfp_tex = make_figure_block(
        dataset="gfp",
        train_sizes=[1000, 2000, 5000, 10000],
        plot_dir=args.plot_dir,
        label="fig:appendix_gfp_methods",
        caption=gfp_caption,
    )

    aav_tex = make_figure_block(
        dataset="aav",
        train_sizes=[1000, 2000, 5000, 10000],
        plot_dir=args.plot_dir,
        label="fig:appendix_aav_methods",
        caption=aav_caption,
    )

    gfp_path = os.path.join(args.output_dir, "appendix_gfp_methods_figure.tex")
    aav_path = os.path.join(args.output_dir, "appendix_aav_methods_figure.tex")
    all_path = os.path.join(args.output_dir, "appendix_all_figures.tex")

    with open(gfp_path, "w", encoding="utf-8") as f:
        f.write(gfp_tex)

    with open(aav_path, "w", encoding="utf-8") as f:
        f.write(aav_tex)

    with open(all_path, "w", encoding="utf-8") as f:
        f.write(rf"\input{{{gfp_path.replace(os.sep, '/')}}}" + "\n\n")
        f.write(rf"\input{{{aav_path.replace(os.sep, '/')}}}" + "\n")

    print(f"Saved: {gfp_path}")
    print(f"Saved: {aav_path}")
    print(f"Saved: {all_path}")
    print()
    print(gfp_tex)
    print()
    print(aav_tex)


if __name__ == "__main__":
    main()
