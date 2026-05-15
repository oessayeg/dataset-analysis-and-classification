import json
import sys
import numpy as np
import pandas as pd

EPOCHS = 10000
LEARNING_RATE = 0.1
TOLERANCE = 1e-6

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


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[FEATURES + [TARGET]].dropna()
    return df


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compute_loss(y: np.ndarray, y_hat: np.ndarray) -> float:
    n = len(y)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / n


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def save_weights(weights: dict[str, dict[str, np.ndarray | float]]):
    model = {
        house: {"w": data["w"].tolist(), "b": float(data["b"])}  # type: ignore[union-attr]
        for house, data in weights.items()
    }
    with open("weights.json", "w") as f:
        json.dump(model, f)
    print("Weights saved to weights.json")


def train(df: pd.DataFrame):
    houses = sorted(df[TARGET].unique())
    X = df[FEATURES].values
    X, mean, std = standardize(X)
    n_features = X.shape[1]

    weights = {}
    for house in houses:
        y = (df[TARGET] == house).astype(int).values
        w = np.zeros(n_features)
        b = 0.0

        loss = float("inf")
        prev_loss = float("inf")

        for epoch in range(EPOCHS):
            z = X @ w + b
            y_hat = sigmoid(z)

            loss = compute_loss(y, y_hat)
            if abs(prev_loss - loss) < TOLERANCE:
                print(f"{house}: converged at epoch {epoch}")
                break
            prev_loss = loss

            error = y_hat - y
            dw = (X.T @ error) / len(y)
            db = np.sum(error) / len(y)

            w -= LEARNING_RATE * dw
            b -= LEARNING_RATE * db

        print(f"{house}: final loss = {loss:.4f}")
        w_real = w / std
        b_real = b - np.sum(w * mean / std)
        weights[house] = {"w": w_real, "b": b_real}

    save_weights(weights)


def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)

    df = load_dataset(sys.argv[1])
    print(f"Dataset loaded: {len(df)} samples, {len(FEATURES)} features")
    train(df)


if __name__ == "__main__":
    main()
