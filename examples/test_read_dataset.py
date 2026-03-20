import numpy as np


def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["items"]


def main():
    dataset_path = "examples/synthetic_peptides.npz"
    X, y, items = load_dataset(dataset_path)

    print("Loaded:", dataset_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("num items:", len(items))

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(items)

    print("\nFirst 3 examples:")
    for i in range(min(3, len(items))):
        print(f"{i}: item={items[i]} y={y[i]:.6f} x={X[i]}")

    print("\nDataset test passed.")


if __name__ == "__main__":
    main()
