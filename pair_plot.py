import pandas as pd
import matplotlib.pyplot as plt
from constants import DATASET_FILENAME


def show_pair_plot():
    df = pd.read_csv(DATASET_FILENAME)
    houses: list[str] = sorted(df["Hogwarts House"].dropna().unique().tolist())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    house_colors = {
        house: color_cycle[i % len(color_cycle)] for i, house in enumerate(houses)
    }

    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col != "Index"
    ]

    n = len(numeric_cols)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))

    for row, col_y in enumerate(numeric_cols):
        for col, col_x in enumerate(numeric_cols):
            ax = axes[row][col]
            ax.tick_params(labelsize=5)

            if row == col:
                for house in houses:
                    data = df[df["Hogwarts House"] == house][col_x].dropna()
                    ax.hist(data, bins=20, alpha=0.5, color=house_colors[house])
            else:
                for house in houses:
                    mask = df["Hogwarts House"] == house
                    ax.scatter(
                        df[mask][col_x],
                        df[mask][col_y],
                        alpha=0.3,
                        s=1,
                        color=house_colors[house],
                    )

            if row == n - 1:
                ax.set_xlabel(col_x, fontsize=6, rotation=45, ha="right")
            if col == 0:
                ax.set_ylabel(col_y, fontsize=6, rotation=45, ha="right")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=house_colors[h],
            markersize=6,
            label=h,
        )
        for h in houses
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=7)
    fig.suptitle("Pair Plot — Hogwarts Houses", fontsize=12)
    plt.tight_layout(rect=(0, 0, 0.95, 0.97))
    plt.show()


if __name__ == "__main__":
    show_pair_plot()
