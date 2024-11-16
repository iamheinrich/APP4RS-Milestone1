from working_with_tabular_data.tabular_operations import print_counts_per_season, print_avg_num_labels, print_max_num_labels
from working_with_geospatial_vector_data.geo_parquet_operations import print_avg_num_labels as print_avg_num_labels_geo, print_num_overlapping_patches
from creating_splits_for_dl.create_splits import save_splits_to_csv


def main():
    # Task 3: Working with tabular data
    path = "./untracked-files/milestone01/metadata.parquet"
    print_counts_per_season(path)
    print_avg_num_labels(path)
    print_max_num_labels(path)

    # Task 5: Working with geospatial vector data
    file_path = "./untracked-files/milestone01/geoparquets/"
    print_avg_num_labels_geo(file_path)
    print_num_overlapping_patches(file_path)

    # Task 6: Creating train/test splits for deep learning
    metadata_path = "./untracked-files/milestone01/metadata.parquet"
    save_splits_to_csv(
        metadata_path, output_path="./untracked-files/split.csv")


if __name__ == "__main__":
    main()
