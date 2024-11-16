import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metadata(path: str) -> pd.DataFrame:
    full_path = os.path.abspath(path)
    assert os.path.exists(full_path), f"File does not exist: {full_path}"
    metadata = pd.read_parquet(full_path)
    return metadata


def get_tile_id(patch_id: str) -> str:
    parts = patch_id.split("_")
    assert len(parts) == 8, f"Invalid patch_id format: {patch_id}"
    tile_id = parts[5]
    assert tile_id.startswith(
        "T"), f"Tile ID does not start with 'T': {tile_id}"
    return tile_id


def get_hour_of_day(patch_id: str) -> int:
    parts = patch_id.split("_")
    assert len(parts) == 8, f"Invalid patch_id format: {patch_id}"
    date_time_part = parts[2]
    date_time_parts = date_time_part.split("T")
    assert len(
        date_time_parts) == 2, f"Invalid date-time format in patch_id: {patch_id}"
    hour_of_day_str = date_time_parts[1][:2]
    assert hour_of_day_str.isdigit() and len(
        hour_of_day_str) == 2, f"Hour of day is not in the correct format: {hour_of_day_str}"
    return int(hour_of_day_str)


def extract_h_order(patch_id: str) -> int:
    parts = patch_id.split('_')
    assert len(parts) == 8, f"Invalid patch_id format for H order: {patch_id}"
    h_order_str = parts[-2]
    assert h_order_str.isdigit(), f"H order is not a digit: {h_order_str}"
    return int(h_order_str)


def extract_v_order(patch_id: str) -> int:
    parts = patch_id.split('_')
    assert len(parts) == 8, f"Invalid patch_id format for V order: {patch_id}"
    v_order_str = parts[-1]
    assert v_order_str.isdigit(), f"V order is not a digit: {v_order_str}"
    return int(v_order_str)


def create_tile_id_column(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata['tile_id'] = metadata['patch_id'].apply(get_tile_id)
    return metadata


def plot_distribution_of_time(metadata: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    hours = metadata['patch_id'].apply(get_hour_of_day)
    # range(25) creates edges at 0,1,2,...,24
    plt.hist(hours, bins=range(25), align='left', edgecolor='black')
    plt.xticks(range(24))  # Show all 24 hours on x-axis
    plt.xlabel('Hour of Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time of Day')
    plt.tight_layout()


def split_train_test(metadata: pd.DataFrame, test_ratio: float = 0.2) -> pd.DataFrame:
    metadata = create_tile_id_column(metadata)
    metadata = metadata.copy()
    metadata['H'] = metadata['patch_id'].apply(extract_h_order)
    metadata['V'] = metadata['patch_id'].apply(extract_v_order)
    metadata['split'] = 'train'

    central_width_factor = np.sqrt(test_ratio)

    grouped = metadata.groupby('tile_id')

    for tile_id, group in grouped:
        min_H, max_H = group['H'].min(), group['H'].max()
        min_V, max_V = group['V'].min(), group['V'].max()

        range_H = max_H - min_H + 1
        range_V = max_V - min_V + 1

        central_width_H = max(int(central_width_factor * range_H), 1)
        central_width_V = max(int(central_width_factor * range_V), 1)

        central_min_H = min_H + (range_H - central_width_H) // 2
        central_max_H = central_min_H + central_width_H

        central_min_V = min_V + (range_V - central_width_V) // 2
        central_max_V = central_min_V + central_width_V

        condition = (
            (metadata['tile_id'] == tile_id) &
            (metadata['H'] >= central_min_H) & (metadata['H'] < central_max_H) &
            (metadata['V'] >= central_min_V) & (metadata['V'] < central_max_V)
        )

        metadata.loc[condition, 'split'] = 'test'

    return metadata


def plot_split_distribution(metadata: pd.DataFrame):
    # Calculate counts and percentages
    split_counts = metadata['split'].value_counts()
    total_patches = len(metadata)
    split_percentages = (split_counts / total_patches * 100).round(1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute numbers plot
    split_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Number of Patches per Split')
    ax1.set_ylabel('Number of Patches')

    # Add value labels on bars
    for i, v in enumerate(split_counts):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    # Percentage plot
    split_percentages.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
    ax2.set_title('Percentage of Patches per Split')
    ax2.set_ylabel('Percentage (%)')

    # Add percentage labels on bars
    for i, v in enumerate(split_percentages):
        ax2.text(i, v, f'{v}%', ha='center', va='bottom')

    # Adjust layout
    for ax in [ax1, ax2]:
        ax.set_xlabel('Split')
        plt.setp(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()


def plot_label_distribution(metadata: pd.DataFrame):
    train_data = metadata[metadata['split'] == 'train']
    test_data = metadata[metadata['split'] == 'test']

    def count_labels(df):
        all_labels = df['labels'].explode()
        return all_labels.value_counts()

    train_counts = count_labels(train_data)
    test_counts = count_labels(test_data)

    label_df = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).fillna(0)

    plt.figure(figsize=(12, 6))
    label_df.plot(kind='bar')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution in Train vs Test Data')
    plt.legend(title='Split')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


def save_splits_to_csv(metadata_path: str, output_path: str = "./untracked-files/split.csv"):
    # Load the metadata
    metadata = load_metadata(metadata_path)

    # Create train/test split
    metadata = split_train_test(metadata)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a new DataFrame with patch_ids in their respective columns
    train_patches = metadata[metadata['split'] ==
                             'train']['patch_id'].reset_index(drop=True)
    test_patches = metadata[metadata['split'] ==
                            'test']['patch_id'].reset_index(drop=True)

    split_df = pd.DataFrame({
        'train': train_patches,
        'test': test_patches
    })

    # Save to CSV
    split_df.to_csv(output_path, index=False)
    print(f"Created csv file with train/test split at {output_path}")


# Uncomment to plot the distribution of time, label distribution and split distribution

# def main():
#     path = "./untracked-files/milestone01/metadata.parquet"
#     metadata = load_metadata(path)
#     metadata = split_train_test(metadata)
#     plot_distribution_of_time(metadata)
#     plot_label_distribution(metadata)
#     plot_split_distribution(metadata)
#     plt.show()


# if __name__ == "__main__":
#     main()
