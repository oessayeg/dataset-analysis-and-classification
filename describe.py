import pandas as pd
from math import sqrt

DATASET_FILENAME = "dataset_train.csv"


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
    first_segment_of_split_column = sorted_column[
        : int(((len(sorted_column) - 1) / 2))
    ]
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


def descrine_column(column: pd.Series):
    full_count = get_count(column)
    full_count_excluding_nan_values = get_count_excluding_nan_values(column)
    mean = get_mean(column)
    min_value = get_minimum_value(column)
    max_value = get_maximum_value(column)
    standard_deviation = get_standard_deviation(
        mean, full_count_excluding_nan_values, column
    )
    q1, q2, q3 = get_quartiles(column)

    print("Full count ->", full_count)
    print("Count excluding nan values ->", full_count_excluding_nan_values)
    print("Mean ->", mean)
    print("Standard deviation ->", standard_deviation)
    print("Minimum value ->", min_value)
    print("Maximum value ->", max_value)
    print("Q1 ->", q1)
    print("Q2 ->", q2)
    print("Q3 ->", q3)


if __name__ == "__main__":
    df = read_csv_dataset(DATASET_FILENAME)

    for column in df.columns:
        is_column_numeric = pd.api.types.is_numeric_dtype(df[column])

        if is_column_numeric:
            print(f"---- {column} ----")
            descrine_column(df[column])
        else:
            continue
