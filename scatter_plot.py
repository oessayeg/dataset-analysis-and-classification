import pandas as pd
import matplotlib.pyplot as plt
from constants import DATASET_FILENAME


def show_scatter_plot():
    df = pd.read_csv(DATASET_FILENAME)
    houses: list[str] = sorted(df["Hogwarts House"].dropna().unique().tolist())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    house_colors = {
        house: color_cycle[i % len(color_cycle)] for i, house in enumerate(houses)
    }

    _, ax = plt.subplots(figsize=(8, 6))

    for house in houses:
        mask = df["Hogwarts House"] == house
        ax.scatter(
            df[mask]["Astronomy"],
            df[mask]["Defense Against the Dark Arts"],
            alpha=0.5,
            s=10,
            color=house_colors[house],
            label=house,
        )

    ax.set_xlabel("Astronomy")
    ax.set_ylabel("Defense Against the Dark Arts")
    ax.set_title("Astronomy vs Defense Against the Dark Arts")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_scatter_plot()
