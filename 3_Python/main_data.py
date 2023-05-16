from src_data.merge_datasets import get_frames_from_dataset, merge_frames_from_dataset

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    # --- Settings of Data Processing
    path2file = "data"
    # align mode = 0: no aligning, 1: maximum, 2: minimum, 3: maximum positive slop, 4: maximum negative slope
    get_frames_from_dataset(
        path2save=path2file,
        data_set=2,
        align_mode=0
    )
    # --- Merging the frames to new cluster device
    merge_frames_from_dataset()
