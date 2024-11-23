import pytest
import pandas as pd
from working_with_tabular_data.tabular_operations import (
    determine_season_from_patch_id,
    count_rows_per_season,
    add_season_column_to_metadata
)
# Test data


@pytest.fixture
def sample_metadata():
    return pd.DataFrame({
        'patch_id': [
            'S2A_MSIL2A_20230115T000000_N0509_R123_T123_20230115T123456',  # winter
            'S2A_MSIL2A_20230415T000000_N0509_R123_T123_20230415T123456',  # spring
            'S2A_MSIL2A_20230715T000000_N0509_R123_T123_20230715T123456',  # summer
            'S2A_MSIL2A_20231015T000000_N0509_R123_T123_20231015T123456',  # autumn
        ],
        'labels': [
            ['forest'],
            ['water', 'urban'],
            ['agriculture', 'forest', 'urban'],
            ['water']
        ]
    })


def test_determine_season_from_patch_id():
    # Test all seasons
    winter_id = 'S2A_MSIL2A_20230115T000000_N0509_R123_T123_20230115T123456'
    spring_id = 'S2A_MSIL2A_20230415T000000_N0509_R123_T123_20230415T123456'
    summer_id = 'S2A_MSIL2A_20230715T000000_N0509_R123_T123_20230715T123456'
    autumn_id = 'S2A_MSIL2A_20231015T000000_N0509_R123_T123_20231015T123456'

    assert determine_season_from_patch_id(winter_id) == 'winter'
    assert determine_season_from_patch_id(spring_id) == 'spring'
    assert determine_season_from_patch_id(summer_id) == 'summer'
    assert determine_season_from_patch_id(autumn_id) == 'autumn'


def test_determine_season_from_patch_id_invalid_format():
    # Test invalid format
    invalid_id = 'invalid_format'
    with pytest.raises(IndexError):
        determine_season_from_patch_id(invalid_id)


def test_determine_season_from_patch_id_invalid_date():
    # Test invalid date format
    invalid_date_id = 'S2A_MSIL2A_2023011_N0509_R123_T123_20230115T123456'
    with pytest.raises(AssertionError):
        determine_season_from_patch_id(invalid_date_id)


def test_add_season_column_to_metadata(sample_metadata):
    result = add_season_column_to_metadata(sample_metadata)

    assert 'season' in result.columns
    assert list(result['season']) == ['winter', 'spring', 'summer', 'autumn']


def test_count_rows_per_season(sample_metadata):
    metadata_with_seasons = add_season_column_to_metadata(sample_metadata)
    spring, summer, autumn, winter = count_rows_per_season(
        metadata_with_seasons)

    assert spring == 1
    assert summer == 1
    assert autumn == 1
    assert winter == 1


def test_load_metadata_nonexistent_file():
    with pytest.raises(AssertionError):
        load_metadata("nonexistent_file.parquet")
