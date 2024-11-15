##########
# TASK 4 #
##########

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import pandas as pd
import os
import numpy as np

metadata_df = pd.read_parquet('untracked-files/milestone01/metadata.parquet', engine='pyarrow')


def band_code_to_valid_size(band_code: str) -> int:
    "Derive image resolution from band_code string and then return valid image size value"

    pixels_per_metre = -1

    if band_code in ["B02", "B03", "B04", "B08"]:
        pixels_per_metre = 10
    if band_code in ["B05", "B06", "B07", "B8A", "B11", "B12"]:
        pixels_per_metre = 20
    if band_code in ["B01", "B09"]:
        pixels_per_metre = 60    
    
    assert(-1 != pixels_per_metre) # patch_code must fit into one of the 3 previous groups with specific resolution

    return int(1200 / pixels_per_metre)

def checking_correctness(path: str):
    "Task 4.1"

    wrong_size = 0
    with_no_data = 0
    not_part_of_dataset = 0


    #/untracked-files/milestone01/BigEarthNet-v2.0-S2-with-errors/
    dataset_path = path + "BigEarthNet-v2.0-S2-with-errors/"

    for tile_name in os.listdir(dataset_path):
        tile_path = dataset_path + tile_name + "/"

        for patch_id in os.listdir(tile_path):
            patch_path = tile_path + patch_id + "/"
            
            assert(len(os.listdir(patch_path)) == 12) # patch must have 12 bands, no B10 but B08A

            current_patch_wrong_size = 0
            current_patch_with_no_data = 0
            current_patch_not_part_of_dataset = 0

            if patch_id not in metadata_df["patch_id"].values:
                current_patch_not_part_of_dataset = 1
                

            for band_file_name in os.listdir(patch_path):
                band_path = patch_path + band_file_name

                band_code = band_file_name[-7:-4]
                with rasterio.open(band_path) as band_reader:

                    valid_size = band_code_to_valid_size(band_code)
                    if band_reader.width != valid_size or band_reader.height != valid_size:
                        current_patch_wrong_size = 1

                    if (band_reader.read_masks(1)!=255).any():
                        current_patch_with_no_data = 1

            wrong_size += current_patch_wrong_size
            with_no_data += current_patch_with_no_data
            not_part_of_dataset += current_patch_not_part_of_dataset
            
            
    print("\nwrong-size: ", wrong_size, 
          "\nwith-no-data: ", with_no_data, 
          "\nnot-part-of-dataset: ", not_part_of_dataset)



def count_and_sum(row, path, band_code):
    """ for a single band of a single patch return count of non-NO_DATA pixels 
    and also return sum of their values to enable calculating mean over all pixels per band over all patches"""

    pixel_count = 0
    pixel_sum = 0

    with rasterio.open(path + row["tile"] + "/" + row["patch_id"] + "/" + row["patch_id"] + "_" + band_code + ".tif" ) as band_reader:
        
        pixel_count = (band_reader.read_masks(1)==255).sum()
        pixel_sum = band_reader.read(1, masked = True).sum()

    return pixel_count, pixel_sum


def count_and_sumSqDev(row, path, band_code, average):
    """ for a single band of a single patch return sum of pixels individual deviations to 
    value mean to enable calculating std deviation over all pixels per band over all patches"""

    squaredDeviation = 0

    with rasterio.open(path + row["tile"] + "/" + row["patch_id"] + "/" + row["patch_id"] + "_" + band_code + ".tif" ) as band_reader:

        squaredDeviation = ((band_reader.read(1, masked = True)-average)**2).sum()

    return squaredDeviation


def calculating_image_statistics(path: str):
    "Task 4.2"
    patches_for_stats_df = pd.read_csv(path+'patches_for_stats.csv.gz',compression="gzip")#.head() 
    
    dataset_path = path + "BigEarthNet-v2.0-S2-with-errors/"

    for band_code in ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]:
        
        counts_and_sums = patches_for_stats_df.apply(count_and_sum, axis=1, result_type="expand", args=(dataset_path, band_code))


        pixel_count, pixel_sum = counts_and_sums[0].sum(), counts_and_sums[1].sum()
        
        mean = pixel_sum / pixel_count

        std_dev = np.sqrt(
            patches_for_stats_df.apply(count_and_sumSqDev, axis=1, result_type="expand", args=(dataset_path, band_code, mean))
                .sum() / 
                (pixel_count - 1)
        )
        

        print(band_code, "mean:", round(mean))
        print(band_code, "std-dev:", round(std_dev))


def retiling_images(path: str):
    "Task 4.3 split patch into 4 subpatches while preserving and adapting relevant georeferencing data to subwindows"

    image_path = path + "BigEarthNet-v2.0-S2-with-errors/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_33_29/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_33_29_B02.tif"
    
    os.makedirs("untracked-files/re-tiled/", exist_ok=True)
    
    write_path_suffixless = "untracked-files/re-tiled/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_33_29_B02"
    suffixes = ["_A.tif", "_B.tif", "_C.tif", "_D.tif"]

    with rasterio.open(image_path) as band_reader:

        assert(band_reader.width % 2 == 0 and band_reader.height % 2 == 0) # task specifies "re-tile it into four equally sized"

        height_middle = band_reader.height // 2
        width_middle = band_reader.width // 2
        
        subpatch_windows = [
            Window(0, 0, width_middle, height_middle), #top left
            Window(width_middle, 0, width_middle, height_middle), #top right
            Window(0, height_middle, width_middle, height_middle), #bottom left
            Window(width_middle, height_middle, width_middle, height_middle) #bottom right
        ]

        for sub_suffix, sub_window in zip(suffixes, subpatch_windows):

            with rasterio.open(
                write_path_suffixless + sub_suffix, mode="w",
                width=width_middle, height=height_middle, count=1, dtype=band_reader.dtypes[0],
                crs=band_reader.crs, transform=band_reader.window_transform(sub_window)
            ) as band_writer:
                band_writer.write(arr=band_reader.read(indexes=1, window=sub_window), indexes=1)


if __name__ == "__main__":
    path = "untracked-files/milestone01/"
    checking_correctness(path)
    calculating_image_statistics(path)
    retiling_images(path)