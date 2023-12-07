from src_data.merge_datasets_frames import get_frames_from_dataset, merge_frames_from_dataset, merge_data_from_diff_data


if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    path2file = "data"

    get_frames_from_dataset(
        path2save=path2file,
        cluster_class_avai=True,
        process_points=[]
    )

    merge_data_from_diff_data(path2file)

    # --- Merging the frames to new cluster device
    merge_frames_from_dataset()
