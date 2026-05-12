import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from constants import DATASET_FILENAME, Q1_KEY, Q3_KEY, MIN_KEY, MAX_KEY, COUNT_KEY
from describe import get_full_columns_description_map


def freedman_diaconis(q1: float, q3: float, n: int) -> float:
    iqr = q3 - q1
    return 2 * iqr * (n ** (-1 / 3))


def build_bin_edges(min_val: float, max_val: float, bin_width: float) -> list[float]:
    edges: list[float] = []
    current = min_val
    while current <= max_val:
        edges.append(current)
        current += bin_width
    if edges[-1] < max_val:
        edges.append(edges[-1] + bin_width)
    return edges


def show_courses_histograms_by_house():
    df = pd.read_csv(DATASET_FILENAME)
    columns_stats = get_full_columns_description_map()
    houses: list[str] = sorted(df["Hogwarts House"].dropna().unique().tolist())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    house_colors = {
        house: color_cycle[i % len(color_cycle)] for i, house in enumerate(houses)
    }

    n_courses = len(columns_stats)
    n_cols = 4
    n_rows = ceil(n_courses / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), squeeze=False
    )
    axes = axes.flatten()

    for i, (course, stats) in enumerate(columns_stats.items()):
        bin_width = freedman_diaconis(
            stats[Q1_KEY], stats[Q3_KEY], int(stats[COUNT_KEY])
        )
        bins = build_bin_edges(stats[MIN_KEY], stats[MAX_KEY], bin_width)

        ax = axes[i]
        for house in houses:
            house_data = df[df["Hogwarts House"] == house][course].dropna().tolist()
            ax.hist(
                house_data,
                bins=bins,
                alpha=0.5,
                label=house,
                color=house_colors[house],
            )

        ax.set_title(course, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for j in range(n_courses, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Score distributions by Hogwarts House", fontsize=14)
    plt.tight_layout(h_pad=3.5, w_pad=2.0, rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    show_courses_histograms_by_house()
