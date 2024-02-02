from src_data.merge_datasets_frames import get_frames_from_dataset, merge_frames_from_dataset, merge_data_from_diff_data
from package.data_merge.merge_utah_decoder import DatasetDecoder


if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    merge_dataset_mode = 1
    path2file = "data"

    match merge_dataset_mode:
        case 0:
            # Merging frames
            get_frames_from_dataset(
                path2save=path2file,
                cluster_class_avai=True,
                process_points=[]
            )

            merge_data_from_diff_data(path2file)

            # --- Merging the frames to new cluster device
            merge_frames_from_dataset()
        case 1:
            # Decoding datset
            merge_handler = DatasetDecoder(path2file)
            merge_handler.generateDataset('./../..')
