from denspp.offline import YamlConfigHandler
from denspp.offline.data_call import SettingsData, DefaultSettingsData
from denspp.offline.pipeline.pipeline_cmds import ProcessingData, SettingsThread, RecommendedSettingsThread


def start_pipeline_processing(object_dataloader, object_pipeline) -> None:
    """Function for handling the processing pipeline
    :param object_dataloader:   object dataloader
    :param object_pipeline:     object pipeline
    """
    # --- Calling YAML config handler
    settings_data, settings_thr = read_yaml_pipeline_config()

    # ----- Preparation: Module calling -----
    print("\nRunning framework for end-to-end neural signal processing (DeNSPP)"
          "\n\nStep #1: Loading data"
          "\n=================================================")
    datahand = object_dataloader(settings_data)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    if settings_data.do_mapping:
        datahand.do_mapping()

    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    print("\nStep #2: Processing data"
          "\n=================================================")
    thr_station = ProcessingData(object_pipeline, settings_thr, dataIn.data_raw, dataIn.electrode_id)
    thr_station.do_processing()

    # --- Plot all plots and save results
    print("\nStep #3: Saving results and plotting"
          "\n=================================================")
    thr_station.do_save_results()
    thr_station.do_plot_results()


def read_yaml_pipeline_config(
        yaml_data_index: str = 'Config_PipelineData',
        yaml_pipe_index: str = 'Config_Pipeline',
    ) -> [SettingsData, SettingsThread]:
    """
    Function for reading/generating the yaml configuration files for getting the transient data and pipeline processing
    :param yaml_data_index: Index with name for reading the yaml configuration file for data loading
    :param yaml_pipe_index: Index with name for reading the yaml configuration file for pipeline processing
    :return:                Classes for handling the data (SettingsDATA) and pipeline processor (SettingsThread)
    """
    yaml_data = YamlConfigHandler(DefaultSettingsData, yaml_name=yaml_data_index)
    settings_data = yaml_data.get_class(SettingsData)

    yaml_threads = YamlConfigHandler(RecommendedSettingsThread, yaml_name=yaml_pipe_index)
    settings_thr = yaml_threads.get_class(SettingsThread)

    return settings_data, settings_thr
