import duckdb

# Load the geoparquet files using duckdb
conn = duckdb.connect(database=':memory:')
conn.execute("INSTALL spatial;")
conn.execute("LOAD spatial;")
QUERY_LABEL_STATS = """
    WITH 
        valid_labels AS (
            SELECT COUNT(*) AS num_labels
            FROM read_parquet('untracked-files/milestone01/geoparquets/*.parquet')
            WHERE DN != 999
            ),
        file_counts AS (
            SELECT COUNT(*) AS num_files
            FROM glob('untracked-files/milestone01/geoparquets/*.parquet')
            )
    SELECT 
        num_labels,
        num_files,
        CAST(num_labels AS FLOAT) / num_files AS average_num_labels
    FROM valid_labels, file_counts
"""

# Extract the associated multi-label set omitting the UNLABELED label & calculate the average number per patch
label_stats_df = conn.execute(QUERY_LABEL_STATS).df()
# Print the result in the following format: geom-average-num-labels: AVG rounded to two decimals (9.18)
print(
    f"geom-average-num-labels: {label_stats_df['average_num_labels'][0]:.2f}")

# Utilize the geographical information to count all overlapping patches
# Patches are considered overlapping if they share any interior point.
# For simplicity, we will assume that all geometries use the same coordinate reference system.
# Print the result in the following format: geom-num-overlaps: #overlaps
