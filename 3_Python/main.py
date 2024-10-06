from package.data_call.call_spike_files import DataLoader
from package.data_call.call_handler import RecommendedSettingsDATA
from package.pipeline_cmds import ProcessingData, RecommendedThreadSetting
from package.yaml_handler import yaml_config_handler
from src_neuro.pipeline_v1 import Pipeline


if __name__ == "__main__":
    # --- Calling YAML config handler
    yaml_data = yaml_config_handler(RecommendedSettingsDATA, yaml_name='Config_PipelineData')
    settings_data = yaml_data.get_class('SettingsDATA')
    yaml_threads = yaml_config_handler(RecommendedThreadSetting, yaml_name='Config_Pipeline')
    settings_thr = yaml_threads.get_class('ThreadSettings')

    # ----- Preparation: Module calling -----
    print("\nRunning framework for end-to-end neural signal processing (DeNSPP)"
          "\n\nStep #1: Loading data"
          "\n=================================================")
    datahand = DataLoader(settings_data)
    datahand.do_call()
    # datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    print("\nStep #2: Processing data"
          "\n=================================================")
    thr_station = ProcessingData(Pipeline, settings_thr, dataIn.data_raw, dataIn.electrode_id)
    thr_station.do_processing()

    # --- Plot all plots and save results
    print("\nStep #3: Saving results and plotting"
          "\n=================================================")
    thr_station.do_save_results()
    thr_station.do_save_results(True)
    thr_station.do_plot_results()
