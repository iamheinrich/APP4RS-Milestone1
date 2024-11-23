from pathlib import Path
from shapely.geometry import Point, LineString, Polygon
import shutil
import os
import pytest
import pandas as pd
import geopandas as gpd

from working_with_geospatial_vector_data.geo_parquet_operations import analyze_label_stats_of_geoparquet_files, print_num_overlapping_patches, get_num_overlapping_patches


class_ids = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243,
             244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521, 522, 523, 999]
empty_gdf = gpd.GeoDataFrame(columns=["DN", "geometry", "bbox"])

# Test analyze_label_stats_of_geoparquet_files
# Create several dataframes with different number of label per file.
# geometry and bbox are the same for all dataframes


def create_test_parquet(file_path: Path, dn_values, geometry=None, bbox=None):
    points = geometry if geometry else [
        Point(i, i) for i in range(len(dn_values))]
    bboxes = bbox if bbox else [(i, i, i + 1, i + 1)
                                for i in range(len(dn_values))]

    # Create a GeoDataFrame with specified DN values, geometry, and bbox
    gdf = gpd.GeoDataFrame({
        "DN": dn_values,
        "bbox": bboxes
    }, geometry=points)
    gdf.to_parquet(file_path)


def test_analyze_label_stats_of_geoparquet_files(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with different DN values
    test_cases = {
        "file1.parquet": [111, 112, 999],  # 2 valid labels, 1 unlabeled
        "file2.parquet": [121, 999, 999],  # 1 valid label, 2 unlabeled
        "file3.parquet": [131, 132, 133]   # 3 valid labels, 0 unlabeled
    }

    # Create parquet files based on test cases
    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        create_test_parquet(file_path, dn_values)

    # Call the function to analyze label stats
    label_stats_df = analyze_label_stats_of_geoparquet_files(str(parquet_dir))

    # Expected results:
    # Total valid labels = 2 + 1 + 3 = 6
    # Total files = 3
    # Average num labels = 6 / 3 = 2.0
    assert label_stats_df["num_labels"][0] == 6, "Total number of valid labels should be 6 but is."
    assert label_stats_df["num_files"][0] == 3, "Total number of files should be 3."
    assert label_stats_df["average_num_labels"][0] == 2.0, "Average number of labels should be 2.0."


def test_analyze_label_stats_with_all_unlabeled(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_unlabeled"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with all DN values as 999 (unlabeled)
    test_cases = {
        "file1.parquet": [999, 999],
        "file2.parquet": [999],
        "file3.parquet": [999, 999, 999]
    }

    # Create parquet files based on test cases
    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        create_test_parquet(file_path, dn_values)

    # Call the function to analyze label stats
    label_stats_df = analyze_label_stats_of_geoparquet_files(str(parquet_dir))

    # Expected results:
    # Total valid labels = 0
    # Total files = 3
    # Average num labels = 0 / 3 = 0.0
    assert label_stats_df["num_labels"][0] == 0, "Total number of valid labels should be 0."
    assert label_stats_df["num_files"][0] == 3, "Total number of files should be 3."
    assert label_stats_df["average_num_labels"][0] == 0.0, "Average number of labels should be 0.0."


def test_analyze_label_stats_with_mixed_labels(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_mixed"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with mixed DN values
    test_cases = {
        "file1.parquet": [111, 999, 121],
        "file2.parquet": [131, 999],
        "file3.parquet": [141, 142, 335, 411, 412, 421, 999]
    }

    # Create parquet files based on test cases
    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        create_test_parquet(file_path, dn_values)

    # Call the function to analyze label stats
    label_stats_df = analyze_label_stats_of_geoparquet_files(str(parquet_dir))

    # Expected results:
    # Total valid labels = 2 + 1 + 6 = 9
    # Total files = 3
    # Average num labels = 9 / 3 = 3.0
    assert label_stats_df["num_labels"][0] == 9, "Total number of valid labels should be 9."
    assert label_stats_df["num_files"][0] == 3, "Total number of files should be 3."
    assert label_stats_df["average_num_labels"][0] == 3.0, "Average number of labels should be 3.0."


def test_analyze_label_stats_with_invalid_class_ids(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_invalid_class_ids"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with invalid DN values
    test_cases = {
        "file1.parquet": [111, 112, 999, 1000]
    }

    # Create parquet files based on test cases
    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        create_test_parquet(file_path, dn_values)

    with pytest.raises(AssertionError):
        analyze_label_stats_of_geoparquet_files(str(parquet_dir))


def test_analyze_label_stats_with_invalid_dn_field(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_invalid_dn_field"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Create a test case with a list in the DN field instead of a single integer
    points = [Point(0, 0)]
    bboxes = [(0, 0, 1, 1)]

    # Create a DataFrame with an invalid DN value (a list instead of an integer)
    gdf = gpd.GeoDataFrame({
        # Invalid: DN contains a list instead of a single integer
        "DN": [[111, 112]],
        "bbox": bboxes
    }, geometry=points)

    file_path = parquet_dir / "file1.parquet"
    gdf.to_parquet(file_path)

    # The function should raise an AssertionError due to invalid DN field
    with pytest.raises(AssertionError):
        analyze_label_stats_of_geoparquet_files(str(parquet_dir))


def test_print_num_overlapping_patches_no_overlap(tmp_path, capsys):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_no_overlap"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with non-overlapping geometries
    test_cases = {
        "file1.parquet": [111],
        "file2.parquet": [112],
        "file3.parquet": [121]
    }

    # Create parquet files with Point geometries that do not overlap
    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        # Create Points at distinct locations
        points = [Point(i, i) for i in range(len(dn_values))]
        # Create GeoDataFrame without bbox
        gdf = gpd.GeoDataFrame({
            "DN": dn_values
        }, geometry=points)
        gdf.to_parquet(file_path)

    # Call the function to analyze overlaps
    print_num_overlapping_patches(str(parquet_dir))

    # Capture the printed output
    captured = capsys.readouterr()

    # Assert that the number of overlaps is 0
    assert "geom-num-overlaps: 0" in captured.out


def test_num_overlapping_patches_with_overlaps(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_with_overlaps"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with overlapping geometries
    test_cases = {
        "file1.parquet": [111],
        "file2.parquet": [112],
        "file3.parquet": [121],
        "file4.parquet": [131, 132],       # two DNs in one file
        "file5.parquet": [133, 134, 135],  # three DNs in one file
        "file6.parquet": [141, 142]        # two DNs in another file
    }

    # Create a common Polygon that overlaps in all files
    overlapping_polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])

    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        # Assign the overlapping polygon to each file
        geometries = [overlapping_polygon for _ in dn_values]
        # Create GeoDataFrame without bbox
        gdf = gpd.GeoDataFrame({
            "DN": dn_values
        }, geometry=geometries)
        gdf.to_parquet(file_path)

    # Call the function to analyze overlaps
    num_overlaps = get_num_overlapping_patches(str(parquet_dir))

    # Since all patches overlap with each other, number of overlaps is C(n,2) where n=6
    expected_overlaps = 15  # C(6,2) = 15
    assert num_overlaps == expected_overlaps, f"Expected {
        expected_overlaps} overlaps, but got {num_overlaps}."


def test_num_overlapping_patches_mixed_geometries(tmp_path):
    # Create a temporary directory to store parquet files
    parquet_dir = tmp_path / "geoparquets_mixed_geometries"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases with mixed geometries: points, lines, polygons
    test_cases = {
        "file1.parquet": [111],
        "file2.parquet": [112],
        "file3.parquet": [121],
        "file4.parquet": [131],
    }

    # Define different geometries
    geometries = [
        Point(0, 0),
        LineString([(1, 1), (2, 2)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        Point(0.5, 0.5)  # This point overlaps with the polygon
    ]

    for filename, dn_values in test_cases.items():
        file_path = parquet_dir / filename
        # Assign geometries based on filename
        if filename == "file1.parquet":
            g = geometries[0]
        elif filename == "file2.parquet":
            g = geometries[1]
        elif filename == "file3.parquet":
            g = geometries[2]
        elif filename == "file4.parquet":
            g = geometries[3]
        # Create GeoDataFrame without bbox
        gdf = gpd.GeoDataFrame({
            "DN": dn_values
        }, geometry=[g])
        gdf.to_parquet(file_path)

    # Call the function to analyze overlaps
    num_overlaps = get_num_overlapping_patches(str(parquet_dir))

    # Expected overlaps:
    # file1 (Point(0,0)) overlaps with file3 (Polygon covering (0,0)-(1,1))
    # file4 (Point(0.5,0.5)) overlaps with file3 (Polygon) as well
    # So total overlaps: file1 & file3, file4 & file3
    expected_overlaps = 2
    assert num_overlaps == expected_overlaps, f"Expected {
        expected_overlaps} overlaps, but got {num_overlaps}."
