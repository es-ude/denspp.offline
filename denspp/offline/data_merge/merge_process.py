from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.data_call.call_handler import SettingsDATA, RecommendedSettingsDATA
from denspp.offline.data_merge.merge_datasets_frames import MergeDatasets


def start_merge_process(data_loader, pipeline) -> None:
    """Function for preparing and starting the merge process for generating datasets
    :param data_loader:     DataLoader object
    :param pipeline:        Pipeline object
    """
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    yaml_handler = YamlConfigHandler(
        yaml_template=RecommendedSettingsDATA,
        path2yaml='config',
        yaml_name='Config_Merge'
    )
    settings = yaml_handler.get_class(SettingsDATA)

    # ---- Merging spike frames from several files to one file


    merge_handler = MergeDatasets(pipeline, settings, True)
    merge_handler.get_frames_from_dataset(
        data_loader=data_loader,
        cluster_class_avai=False,
        process_points=[]
    )
    merge_handler.merge_data_from_diff_files()
    merge_handler.save_merged_data_in_npyfile()

    # --- Merging the frames to new cluster device
    print("\n====================================================="
          "\nFinal Step with merging cluster have to be done in MATLAB")

