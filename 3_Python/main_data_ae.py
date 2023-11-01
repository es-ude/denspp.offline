from src_data.merge_datasets_frames import get_frames_from_dataset_unique, merge_frames_from_dataset

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    path2file = "data"
    get_frames_from_dataset_unique(
        path2save=path2file,
        cluster_class_avai=True,
        process_points=[2, 10]
    )
    # --- Merging the frames to new cluster device
    merge_frames_from_dataset()
