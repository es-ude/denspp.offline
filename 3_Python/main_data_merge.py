from src_data.merge_datasets_frames import MergeDatasets, merge_frames_from_dataset
from package.merge.merge_datasets import SortDataset

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    path2file = "data"

    merge_dataset = MergeDatasets(path2save=path2file)
    merge_dataset.get_frames_from_dataset(
        cluster_class_avai=True,
        process_points=[]
    )
    merge_dataset.merge_data_from_diff_data()
    path2mergedfile = merge_dataset.get_filepath()
    sort_dataset = SortDataset(path_2_file=path2mergedfile)
    sort_dataset.sort_dataset()
