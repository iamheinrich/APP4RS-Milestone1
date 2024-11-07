from working_with_tabular_data.tabular_operations import print_counts_per_season, print_avg_num_labels, print_max_num_labels


def main():
    # Task: Working with tabular data
    path = "./untracked-files/milestone01/metadata.parquet"
    print_counts_per_season(path)
    print_avg_num_labels(path)
    print_max_num_labels(path)

    # Task: Working with geospatial vector data


if __name__ == "__main__":
    main()
