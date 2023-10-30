from src_data.merge_datasets import get_frames_from_dataset, merge_frames_from_dataset

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    # --- Settings of Data Processing
    path2file = "data"

    get_frames_from_dataset(
        path2save=path2file,
        cluster_class_avai=True,
        process_points=[0, 1],
        do_step_save=True
    )
    # --- Merging the frames to new cluster device
    merge_frames_from_dataset()
