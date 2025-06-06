from logging import getLogger
from denspp.offline.yaml_handler import YamlHandler
from denspp.offline.data_call import SettingsData, DefaultSettingsData
from denspp.offline.pipeline.pipeline_cmds import ProcessingData, SettingsThread, RecommendedSettingsThread


def start_pipeline_processing(object_dataloader, object_pipeline) -> None:
    """Function for handling the processing pipeline
    :param object_dataloader:   object dataloader
    :param object_pipeline:     object pipeline
    """
    logger = getLogger(__name__)

    # --- Calling YAML config handler
    settings_data, settings_thr = read_yaml_pipeline_config()

    # ----- Preparation: Module calling -----
    logger.info("Running framework for end-to-end neural signal processing (DeNSPP)")
    logger.info("Step #1: Loading data")
    logger.info("==================================================================")

    datahand = object_dataloader(settings_data)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.do_mapping()

    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    logger.info("Step #2: Processing data")
    logger.info("==================================================================")
    thr_station = ProcessingData(
        pipeline=object_pipeline,
        settings=settings_thr,
        data_in=dataIn.data_raw,
        fs=dataIn.data_fs_used,
        channel_id=dataIn.electrode_id
    )
    thr_station.do_processing()

    # --- Plot all plots and save results
    logger.info("Step #3: Saving results and plotting")
    logger.info("==================================================================")
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
    yaml_data = YamlHandler(DefaultSettingsData, file_name=yaml_data_index)
    settings_data = yaml_data.get_class(SettingsData)

    yaml_threads = YamlHandler(RecommendedSettingsThread, file_name=yaml_pipe_index)
    settings_thr = yaml_threads.get_class(SettingsThread)

    return settings_data, settings_thr
