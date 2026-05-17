import json
import sys
import numpy as np
import pandas as pd

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


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def load_weights(path: str) -> dict[str, dict[str, list[float] | float]]:
    with open(path) as f:
        return json.load(f)


def predict(
    df: pd.DataFrame, weights: dict[str, dict[str, list[float] | float]]
) -> list[str]:
    X = df[FEATURES].values
    houses = list(weights.keys())
    probabilities = np.zeros((len(X), len(houses)))

    for i, house in enumerate(houses):
        w = np.array(weights[house]["w"])
        b = float(weights[house]["b"])  # type: ignore[arg-type]
        probabilities[:, i] = sigmoid(X @ w + b)

    predicted_indices = np.argmax(probabilities, axis=1)
    return [houses[i] for i in predicted_indices]


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python logreg_predict.py <dataset_path> [weights.json]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    weights_path = sys.argv[2] if len(sys.argv) == 3 else "weights.json"
    weights = load_weights(weights_path)

    imputation_means = weights.pop("imputation_means", {})
    if imputation_means:
        df[FEATURES] = df[FEATURES].fillna(imputation_means)

    predictions = predict(df, weights)
    df[TARGET] = predictions
    df[["Index", TARGET]].to_csv("houses.csv", index=False)
    print(f"Predictions saved to houses.csv ({len(predictions)} students)")


if __name__ == "__main__":
    main()
