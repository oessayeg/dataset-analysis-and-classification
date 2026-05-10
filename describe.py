import pandas as pd
from math import sqrt
from constants import (
    DATASET_FILENAME,
    INDEX_COLUMN,
    FULL_COUNT_KEY,
    COUNT_KEY,
    MEAN_KEY,
    STD_KEY,
    MIN_KEY,
    MAX_KEY,
    Q1_KEY,
    Q2_KEY,
    Q3_KEY,
)


def read_csv_dataset(filename: str):
    return pd.read_csv(filename)


def get_count(column: pd.Series) -> int:
    return len(column)


def get_count_excluding_nan_values(column: pd.Series) -> int:
    count = 0

    for value in column:
        if not pd.isna(value):
            count += 1
    return count


def get_mean(column: pd.Series) -> float:
    sum_of_values = 0
    number_valid_values = 0

    for value in column:
        if pd.isna(value):
            continue
        sum_of_values += value
        number_valid_values += 1

    return sum_of_values / number_valid_values


def get_minimum_value(column: pd.Series):
    got_first_minimum_value = False
    min_value = 0

    for value in column:
        if pd.isna(value):
            continue
        if not got_first_minimum_value:
            min_value = value
            got_first_minimum_value = True
        if got_first_minimum_value and value < min_value:
            min_value = value

    return min_value


def get_median(sorted_column: pd.Series):
    if len(sorted_column) % 2 == 1:
        return sorted_column.iloc[(len(sorted_column) - 1) // 2]

    return (
        sorted_column.iloc[(len(sorted_column) // 2) - 1]
        + sorted_column.iloc[(len(sorted_column)) // 2]
    ) / 2


def get_first_quartile(sorted_column: pd.Series):
    if len(sorted_column) % 2 == 0:
        first_segment_of_split_column = sorted_column[: int(len(sorted_column) / 2)]
        return get_median(first_segment_of_split_column)
    first_segment_of_split_column = sorted_column[: int(((len(sorted_column) - 1) / 2))]
    return get_median(first_segment_of_split_column)


def get_third_quartile(sorted_column: pd.Series):
    if len(sorted_column) % 2 == 0:
        first_segment_of_split_column = sorted_column[int(len(sorted_column) / 2) :]
        return get_median(first_segment_of_split_column)
    first_segment_of_split_column = sorted_column[
        int(((len(sorted_column) - 1) / 2) + 1) :
    ]
    return get_median(first_segment_of_split_column)


def get_quartiles(column: pd.Series) -> tuple[float, float, float]:
    cleaned_column = column.dropna()
    sorted_column = cleaned_column.sort_values()
    median = get_median(sorted_column)
    first_quartile = get_first_quartile(sorted_column)
    third_quartile = get_third_quartile(sorted_column)

    return first_quartile, median, third_quartile


def get_maximum_value(column: pd.Series):
    got_first_maximum_value = False
    min_value = 0

    for value in column:
        if pd.isna(value):
            continue
        if not got_first_maximum_value:
            min_value = value
            got_first_maximum_value = True
        if got_first_maximum_value and value > min_value:
            min_value = value

    return min_value


def get_standard_deviation(
    mean: float, full_count_excluding_nan_values: int, column: pd.Series
) -> float:
    sum_of_squared_distances_from_mean = 0

    for value in column:
        if pd.isna(value):
            continue
        sum_of_squared_distances_from_mean += (value - mean) ** 2

    return sqrt(
        sum_of_squared_distances_from_mean / (full_count_excluding_nan_values - 1)
    )


def describe_column(column: pd.Series) -> dict[str, int | float]:
    full_count = get_count(column)
    full_count_excluding_nan_values = get_count_excluding_nan_values(column)
    mean = get_mean(column)
    min_value = get_minimum_value(column)
    max_value = get_maximum_value(column)
    standard_deviation = get_standard_deviation(
        mean, full_count_excluding_nan_values, column
    )
    q1, q2, q3 = get_quartiles(column)

    return {
        FULL_COUNT_KEY: full_count,
        COUNT_KEY: full_count_excluding_nan_values,
        MEAN_KEY: mean,
        STD_KEY: standard_deviation,
        MIN_KEY: min_value,
        MAX_KEY: max_value,
        Q1_KEY: q1,
        Q2_KEY: q2,
        Q3_KEY: q3,
    }


def describe(columns_description_map: dict[str, dict[str, int | float]]) -> None:
    row_labels = [
        ("Count", COUNT_KEY),
        ("Mean", MEAN_KEY),
        ("Std", STD_KEY),
        ("Min", MIN_KEY),
        ("25%", Q1_KEY),
        ("50%", Q2_KEY),
        ("75%", Q3_KEY),
        ("Max", MAX_KEY),
    ]

    col_names = list(columns_description_map.keys())
    col_width = 15
    label_width = max(len(label) for label, _ in row_labels)

    header = " " * label_width + "".join(name.rjust(col_width) for name in col_names)
    print(header)

    for label, key in row_labels:
        row = label.ljust(label_width) + "".join(
            f"{columns_description_map[col][key]:>{col_width}.6f}" for col in col_names
        )
        print(row)


def describe_transposed(
    columns_description_map: dict[str, dict[str, int | float]],
) -> None:
    stat_labels = [
        ("Count", COUNT_KEY),
        ("Mean", MEAN_KEY),
        ("Std", STD_KEY),
        ("Min", MIN_KEY),
        ("25%", Q1_KEY),
        ("50%", Q2_KEY),
        ("75%", Q3_KEY),
        ("Max", MAX_KEY),
    ]

    col_names = list(columns_description_map.keys())
    stat_headers = [label for label, _ in stat_labels]
    stat_width = 15
    label_width = max(len(name) for name in col_names)

    header = " " * label_width + "".join(h.rjust(stat_width) for h in stat_headers)
    print(header)

    for col_name in col_names:
        row = col_name.ljust(label_width) + "".join(
            f"{columns_description_map[col_name][key]:>{stat_width}.6f}"
            for _, key in stat_labels
        )
        print(row)


def main():
    df = read_csv_dataset(DATASET_FILENAME)
    full_columns_description_map: dict[str, dict[str, int | float]] = {}

    for column in df.columns:
        is_column_numeric = pd.api.types.is_numeric_dtype(df[column])

        if is_column_numeric and column != INDEX_COLUMN:
            column_description_map = describe_column(df[column])
            full_columns_description_map[column] = column_description_map
        else:
            continue

    # describe(full_columns_description_map)
    describe_transposed(full_columns_description_map)


if __name__ == "__main__":
    main()
