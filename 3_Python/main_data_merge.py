if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    merge_dataset_mode = 0
    path2file = "data"

    match merge_dataset_mode:
        case 0:
            # ---- Merging spike frames from several files to one file
            from package.data_merge.merge_datasets_frames import merge_frames_from_dataset, MergeDatasets
            merge_handler = MergeDatasets(path2file)
            # Merging frames
            merge_handler.get_frames_from_dataset(
                cluster_class_avai=False,
                process_points=[]
            )
            merge_handler.merge_data_from_diff_data()
            # --- Merging the frames to new cluster device
            merge_frames_from_dataset()
        case 1:
            # ---- Merging decoding (KlaesLab) from several files to one file
            from package.data_merge.merge_utah_decoder import DatasetDecoder
            # Decoding datset
            merge_handler = DatasetDecoder(path2file)
            merge_handler.generateDataset('./../..')
        case 2:
            # ---- Merging spike frames from several files to one file for spike detection
            from package.data_merge.merge_datasets_sda import prepare_sda_dataset
            # Spike Detection Algorithm
            prepare_sda_dataset(
                path2save=path2file,
                process_points=[]
            )
