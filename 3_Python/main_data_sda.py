from src_data.merge_datasets_sda import prepare_sda_dataset
from src_data.merge_datasets_frames import merge_frames_from_dataset

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    path2file = "data"
    prepare_sda_dataset(
        path2save=path2file,
        process_points=[]
    )
    # --- Merging the frames to new cluster device
    merge_frames_from_dataset()
