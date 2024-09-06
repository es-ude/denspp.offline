from package.data_call.call_spike_files import DataLoader, SettingsDATA
from package.pipeline_cmds import ThreadSettings, ProcessingData
from src_neuro.pipeline_v1 import Pipeline


if __name__ == "__main__":
    # --- Settings for Processing
    SettingsDATA = SettingsDATA(
        # path='../2_Data',
        path='C:\HomeOffice\Data_Neurosignal',
        data_set=1,
        data_case=0,
        data_point=0,
        t_range=[0],
        ch_sel=[],
        fs_resample=50e3
    )
    settings_thr = ThreadSettings(
        use_multithreading=True,
        num_max_workers=2,
        block_plots=True,
        fs_ana=SettingsDATA.fs_resample,
        pipeline=Pipeline
    )

    # ----- Preparation: Module calling -----
    print("\nRunning framework for end-to-end neural signal processing (DeNSPP)"
          "\n\nStep #1: Loading data"
          "\n=================================================")
    datahand = DataLoader(SettingsDATA)
    datahand.do_call()
    # datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    print("\nStep #2: Processing data"
          "\n=================================================")
    thr_station = ProcessingData(settings_thr, dataIn)
    thr_station.do_processing()

    # --- Plot all plots and save results
    print("\nStep #3: Saving results and plotting"
          "\n=================================================")
    thr_station.do_save_results()
    thr_station.do_save_results(True)
    thr_station.do_plot_results()
