import duckdb
import os


def determine_season_from_patch_id(patch_id: str):
    # The season is calculated for the northern hemisphere
    # The patch_id is in the format <Sentinel-ID>_MSIL2A_<YYYYMMDD>T<HHMMSS>_N9999_<Rooo>_←-<Txxxxxx>_<H-Order>_<V-Order>
    # The season is determined by the month and day of the acquisition, according to https://en.wikipedia.org/wiki/Season#Meteorological

    # Extract date and time as YYYYMMDD
    datetime = patch_id.split("_")[2].split("T")[0]
    assert len(datetime) == 8, f"Datetime is not in the correct format: {
        datetime}, expected YYYYMMDD corresponding to 8 characters"
    # Extract month
    month = int(datetime[4:6])
    assert month <= 12 and month >= 1, f"Month is out of range: {month}"

    # Determine the season
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"


def load_metadata(path: str):
    # Get absolute path
    full_path = os.path.abspath(path)
    # Check if file exists
    assert os.path.exists(full_path), f"File does not exist: {full_path}"
    # Load metadata from parquet file
    metadata = duckdb.sql(f"SELECT * FROM '{full_path}'").df()
    return metadata


def add_season_column_to_metadata(metadata):
    metadata['season'] = metadata['patch_id'].apply(
        determine_season_from_patch_id)
    return metadata


def count_rows_per_season(metadata):
    # Count the number of rows for winter, spring, summer and autumn
    spring_count = (metadata["season"] == "spring").sum()
    summer_count = (metadata["season"] == "summer").sum()
    autumn_count = (metadata["season"] == "autumn").sum()
    winter_count = (metadata["season"] == "winter").sum()
    return spring_count, summer_count, autumn_count, winter_count


def get_label_statistics(metadata_path: str):
    """Load metadata and return a tuple of (label_counts, metadata) for reuse"""
    metadata = load_metadata(metadata_path)
    labels = metadata["labels"]
    label_counts = labels.apply(len)
    return label_counts


def print_counts_per_season(metadata_path: str):
    metadata = load_metadata(metadata_path)
    metadata = add_season_column_to_metadata(metadata)
    spring_count, summer_count, autumn_count, winter_count = count_rows_per_season(
        metadata)
    print(f"spring: {spring_count}\nsummer: {summer_count}\nautumn: {
          autumn_count}\nwinter: {winter_count}")


def print_avg_num_labels(metadata_path: str):
    label_counts = get_label_statistics(metadata_path)
    avg_num_labels = label_counts.mean()  # More efficient than sum/len
    print(
        f"average-num-labels: {round(avg_num_labels, 2)}")


def print_max_num_labels(metadata_path: str):
    label_counts = get_label_statistics(metadata_path)
    print(f"maximum-num-labels: {label_counts.max()}")
