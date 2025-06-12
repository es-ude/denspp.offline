from logging import getLogger
from denspp.offline.yaml_handler import YamlHandler
from denspp.offline.data_call import SettingsData, DefaultSettingsData
from denspp.offline.pipeline.pipeline_cmds import ProcessingData, SettingsThread, RecommendedSettingsThread


def start_pipeline_processing(object_dataloader, object_pipeline, en_testmode: bool=False,
                              sets_load_data: SettingsData=DefaultSettingsData,
                              sets_load_thread: SettingsThread=RecommendedSettingsThread) -> None:
    """Function for handling the processing pipeline
    :param object_dataloader:   object dataloader
    :param object_pipeline:     object pipeline
    :param en_testmode:         Boolean for enabling the testing mode (skip the yaml load part)
    :param sets_load_data:       Dataclass with settings for getting data to analyse
    :param sets_load_thread:     Dataclass with settings for handling the processing threads
    :return:                    None
    """
    logger = getLogger(__name__)

    # --- Calling YAML config handler
    if not en_testmode:
        settings_data = YamlHandler(
            template=sets_load_data,
            path='config',
            file_name='Config_PipelineData'
        ).get_class(SettingsData)
        settings_thr = YamlHandler(
            template=sets_load_thread,
            path='config',
            file_name='Config_Pipeline'
        ).get_class(SettingsThread)
    else:
        settings_data = sets_load_data
        settings_thr = sets_load_thread

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
        fs=dataIn.fs_used,
        channel_id=dataIn.electrode_id
    )
    thr_station.do_processing()

    # --- Plot all plots and save results
    logger.info("Step #3: Saving results and plotting")
    logger.info("==================================================================")
    thr_station.do_save_results()
    thr_station.do_plot_results()
