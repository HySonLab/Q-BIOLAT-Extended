import numpy as np


class MLPRegressorNumpy:
    # Small NumPy MLP baseline for regression on binary latent codes.

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 32),
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 400,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

        dims = [self.input_dim, *self.hidden_dims, 1]
        self.params = {}
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            self.params[f"W{i+1}"] = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.params[f"b{i+1}"] = np.zeros((1, fan_out), dtype=np.float64)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(np.float64)

    def _forward(self, X):
        caches = {"A0": X}
        L = len(self.hidden_dims) + 1
        A = X
        for i in range(1, L):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self._relu(Z)
            caches[f"Z{i}"] = Z
            caches[f"A{i}"] = A
        Z_out = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        caches[f"Z{L}"] = Z_out
        caches[f"A{L}"] = Z_out
        return Z_out, caches

    def fit(self, X, y, verbose=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X.shape[0]
        L = len(self.hidden_dims) + 1

        for epoch in range(self.epochs):
            perm = self.rng.permutation(n)
            X_shuf = X[perm]
            y_shuf = y[perm]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                pred, caches = self._forward(xb)
                diff = pred - yb
                m = xb.shape[0]
                dA = 2.0 * diff / max(m, 1)

                grads = {}
                dZ = dA
                A_prev = caches[f"A{L-1}"] if L > 1 else caches["A0"]
                grads[f"W{L}"] = A_prev.T @ dZ + self.weight_decay * self.params[f"W{L}"]
                grads[f"b{L}"] = np.sum(dZ, axis=0, keepdims=True)

                dA_prev = dZ @ self.params[f"W{L}"].T

                for i in range(L - 1, 0, -1):
                    dZ = dA_prev * self._relu_grad(caches[f"Z{i}"])
                    A_prev = caches["A0"] if i == 1 else caches[f"A{i-1}"]
                    grads[f"W{i}"] = A_prev.T @ dZ + self.weight_decay * self.params[f"W{i}"]
                    grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True)
                    dA_prev = dZ @ self.params[f"W{i}"].T

                for i in range(1, L + 1):
                    self.params[f"W{i}"] -= self.lr * grads[f"W{i}"]
                    self.params[f"b{i}"] -= self.lr * grads[f"b{i}"]

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                loss = np.mean((self.predict(X) - y.ravel()) ** 2)
                print(f"[MLP] epoch={epoch+1:4d} mse={loss:.6f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        pred, _ = self._forward(X)
        return pred.ravel()
