if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in "
          "end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    merge_dataset_mode = 0
    path2file = "data"

    match merge_dataset_mode:
        case 0:
            # ---- Merging spike frames from several files to one file
            from package.data_merge.merge_datasets_frames import MergeDatasets
            from package.data_call.call_handler import SettingsDATA

            setup_data = SettingsDATA(
                # path='../2_Data',
                # path='/media/erbsloeh/ExtremeSSD/0_Invasive',
                path='C:/HomeOffice/Data_Neurosignal',
                data_set=1, data_case=0, data_point=0,
                t_range=[0],
                ch_sel=[],
                fs_resample=50e3
            )

            merge_handler = MergeDatasets(setup_data, path2file, True)
            merge_handler.get_frames_from_dataset(
                cluster_class_avai=False,
                process_points=[]
            )
            merge_handler.merge_data_from_diff_files()
            merge_handler.save_merged_data_in_matfile()
            merge_handler.save_merged_data_in_npyfile()

            # --- Merging the frames to new cluster device
            print("\n====================================================="
                  "\nFinal Step with merging cluster have to be done in MATLAB")
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
