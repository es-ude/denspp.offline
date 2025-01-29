from package.pipeline.pipeline_cmds import ProcessingData, read_yaml_pipeline_config
from package.template.call_template import DataLoader
from package.template.pipeline_v0 import Pipeline


if __name__ == "__main__":
    # --- Calling YAML config handler
    settings_data, settings_thr = read_yaml_pipeline_config()

    # ----- Preparation: Module calling -----
    print("\nRunning framework for end-to-end neural signal processing (DeNSPP)"
          "\n\nStep #1: Loading data"
          "\n=================================================")
    datahand = DataLoader(settings_data)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.do_mapping()
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
    thr_station.do_plot_results()
