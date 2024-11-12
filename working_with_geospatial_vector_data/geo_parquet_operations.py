import duckdb

CLASS_IDS = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242, 243,
             244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521, 522, 523, 999]


def analyze_label_stats_of_geoparquet_files(file_path: str):
    # Load the geoparquet files using duckdb
    conn = duckdb.connect(database=':memory:')
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    # Check if each field in the DN column contains exactly one integer
    QUERY_CHECK_DN_FIELD = f"SELECT DN FROM read_parquet('{
        file_path}/*.parquet')"
    dn_df = conn.execute(QUERY_CHECK_DN_FIELD).df()
    assert all(isinstance(x, int) for x in dn_df['DN'])

    # Check if all class ids from the parquet files are valid
    QUERY_VALIDATE_CLASS_IDS = f"SELECT DISTINCT DN FROM read_parquet('{
        file_path}/*.parquet')"
    class_ids = conn.execute(QUERY_VALIDATE_CLASS_IDS).df()
    class_ids = class_ids['DN'].tolist()
    assert all(class_id in CLASS_IDS for class_id in class_ids)

    # Calculate the average number of labels per file and the total number of files
    QUERY_LABEL_STATS = f"""
        WITH
            valid_labels AS (
                SELECT COUNT(*) AS num_labels
                FROM read_parquet('{file_path}/*.parquet')
                WHERE DN != 999
                ),
            file_counts AS (
                SELECT COUNT(*) AS num_files
                FROM glob('{file_path}/*.parquet')
                )
        SELECT
            num_labels,
            num_files,
            CAST(num_labels AS FLOAT) / num_files AS average_num_labels
        FROM valid_labels, file_counts
    """

    # Extract the associated multi-label set omitting the UNLABELED label & calculate the average number per patch
    label_stats_df = conn.execute(QUERY_LABEL_STATS).df()

    return label_stats_df


def print_avg_num_labels(file_path: str):
    label_stats_df = analyze_label_stats_of_geoparquet_files(file_path)
    print(
        f"geom-average-num-labels: {label_stats_df['average_num_labels'][0]:.2f}")

# Utilize the geographical information to count all overlapping patches
# Patches are considered overlapping if they share any interior point.
# For simplicity, we will assume that all geometries use the same coordinate reference system.
# Print the result in the following format: geom-num-overlaps: #overlaps

# Understand that one parquet file represents one patch.


def read_geoparquet_file(file_path: str):
    # Load the geoparquet files using duckdb
    conn = duckdb.connect(database=':memory:')
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    return conn.execute(f"SELECT * FROM read_parquet('{file_path}')").df()


def print_num_overlapping_patches(file_path: str):
    conn = duckdb.connect(database=':memory:')
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    # Create a table to store unified patches
    conn.execute(
        "CREATE TABLE unified_patches (patch_id INTEGER, unified_geometry GEOMETRY)")

    # Get list of all parquet files
    files = conn.execute(
        f"SELECT * FROM glob('{file_path}/*.parquet')").df()

    # Process each file and store its unified geometry
    for idx, file_path in enumerate(files['file']):
        conn.execute(f"""
            INSERT INTO unified_patches
            SELECT
                {idx} as patch_id,
                ST_Union_Agg(geometry) as unified_geometry
            FROM read_parquet('{file_path}')
        """)

    # Now check for overlaps between unified patches
    overlaps = conn.execute("""
        WITH overlap_counts AS (
            SELECT 
                a.patch_id,
                COUNT(*) as num_overlaps
            FROM unified_patches a
            JOIN unified_patches b ON a.patch_id < b.patch_id
            WHERE ST_Intersects(a.unified_geometry, b.unified_geometry)
            GROUP BY a.patch_id
        )
        SELECT 
            COUNT(*) as total_overlaps,
        FROM overlap_counts
    """).df()

    print(f"geom-num-overlaps: {int(overlaps['total_overlaps'][0])}")
