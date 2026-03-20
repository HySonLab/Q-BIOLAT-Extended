import numpy as np


def read_dataset(path):
    """
    Read synthetic dataset stored as NumPy .npz
    """

    data = np.load(path, allow_pickle=True)

    X = data["X"]        # binary latent codes
    y = data["y"]        # fitness values
    items = data["items"]  # peptide strings

    return X, y, items


def dataset_summary(X, y, items):

    print("===== Dataset Summary =====")

    print("Number of samples:", X.shape[0])
    print("Latent dimension:", X.shape[1])

    print("Fitness mean:", np.mean(y))
    print("Fitness std:", np.std(y))
    print("Fitness min:", np.min(y))
    print("Fitness max:", np.max(y))

    print("\nFirst few items:")
    for i in range(min(5, len(items))):
        print("Item:", items[i])
        print("Latent code:", X[i])
        print("Fitness:", y[i])
        print("-----")


def main():

    dataset_path = "examples/synthetic_peptides.npz"

    print("Loading dataset:", dataset_path)

    X, y, items = read_dataset(dataset_path)

    dataset_summary(X, y, items)

    # consistency checks
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(items)

    print("\nDataset loaded successfully.")


if __name__ == "__main__":
    main()
