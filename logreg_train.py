import json
import sys
import numpy as np
import pandas as pd

EPOCHS = 10000
LEARNING_RATE = 0.1
TOLERANCE = 1e-6
OPTIMIZER = "batch"

FEATURES = [
    "Defense Against the Dark Arts",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying",
]

TARGET = "Hogwarts House"


def load_dataset(path: str) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(path)
    df = df[FEATURES + [TARGET]]
    imputation_means = df[FEATURES].mean().to_dict()
    df[FEATURES] = df[FEATURES].fillna(imputation_means)
    df = df.dropna(subset=[TARGET])
    return df, imputation_means


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compute_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    n = len(y)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / n


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def save_weights(
    weights: dict[str, dict[str, np.ndarray | float]],
    imputation_means: dict[str, float],
):
    model = {
        house: {"w": data["w"].tolist(), "b": float(data["b"])}  # type: ignore[union-attr]
        for house, data in weights.items()
    }
    model["imputation_means"] = imputation_means  # type: ignore[assignment]
    with open("weights.json", "w") as f:
        json.dump(model, f)
    print("Weights saved to weights.json")


def train_batch(X, y, w, b):
    prev_loss = float("inf")
    loss = float("inf")

    for epoch in range(EPOCHS):
        z = X @ w + b
        y_hat = sigmoid(z)

        loss = compute_loss(y, y_hat)
        if abs(prev_loss - loss) < TOLERANCE:
            print(f"  converged at epoch {epoch}")
            break
        prev_loss = loss

        error = y_hat - y
        dw = (X.T @ error) / len(y)
        db = np.sum(error) / len(y)

        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

    return w, b, loss


def train_sgd(X, y, w, b):
    n = len(y)
    loss = float("inf")

    for epoch in range(EPOCHS):
        indices = np.random.permutation(n)
        for i in indices:
            z = X[i] @ w + b
            y_hat = sigmoid(z)
            error = y_hat - y[i]
            w -= LEARNING_RATE * X[i] * error
            b -= LEARNING_RATE * error

        z = X @ w + b
        y_hat = sigmoid(z)
        prev_loss = loss
        loss = compute_loss(y, y_hat)

        if abs(prev_loss - loss) < TOLERANCE:
            print(f"  converged at epoch {epoch}")
            break

    return w, b, loss


def train(df: pd.DataFrame, imputation_means: dict[str, float]):
    houses = sorted(df[TARGET].unique())
    X = df[FEATURES].values
    X, mean, std = standardize(X)
    n_features = X.shape[1]

    weights = {}
    for house in houses:
        y = (df[TARGET] == house).astype(int).values
        w = np.zeros(n_features)
        b = 0.0

        print(f"Training {house} ({OPTIMIZER})...")
        if OPTIMIZER == "sgd":
            w, b, loss = train_sgd(X, y, w, b)
        else:
            w, b, loss = train_batch(X, y, w, b)

        print(f"  final loss = {loss:.4f}")
        w_real = w / std
        b_real = b - np.sum(w * mean / std)
        weights[house] = {"w": w_real, "b": b_real}

    save_weights(weights, imputation_means)


def main():
    global OPTIMIZER

    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <dataset_path> [--optimizer batch|sgd]")
        sys.exit(1)

    args = sys.argv[1:]
    dataset_path = args[0]

    if "--optimizer" in args:
        idx = args.index("--optimizer")
        if idx + 1 >= len(args):
            print("Error: --optimizer requires a value (batch or sgd)")
            sys.exit(1)
        OPTIMIZER = args[idx + 1]
        if OPTIMIZER not in ("batch", "sgd"):
            print(f"Error: unknown optimizer '{OPTIMIZER}', choose batch or sgd")
            sys.exit(1)

    df, imputation_means = load_dataset(dataset_path)
    print(f"Dataset loaded: {len(df)} samples, {len(FEATURES)} features")
    train(df, imputation_means)


if __name__ == "__main__":
    main()
