from logging import getLogger
from denspp.offline.logger import define_logger_runtime
from denspp.offline.yaml_handler import YamlHandler
from denspp.offline.data_call import SettingsData, DefaultSettingsData
from denspp.offline.data_process.merge_datasets_frames import MergeDatasets
from denspp.offline.pipeline.pipeline_cmds import ProcessingData, SettingsThread, RecommendedSettingsThread, DataloaderLibrary, PipelineLibrary


def start_processing_pipeline(sets_load_data: SettingsData=DefaultSettingsData,
                              sets_load_thread: SettingsThread=RecommendedSettingsThread):
    """Function for preparing the pipeline preprocessing
    :param sets_load_data:      Dataclass with settings for getting data to analyse
    :param sets_load_thread:    Dataclass with settings for handling the processing threads
    :return:                    Tuple with (0) Settings class for data, (1) Settings class for thread handling, (2) DataLoader, (3) Pipeline, (4) mode
    """
    # --- Getting the YAML files
    logger = getLogger(__name__)
    if not sets_load_data == DefaultSettingsData:
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
    logger.debug("Load YAML configs")

    # --- Getting the Pipeline and DataLoader
    datalib = DataloaderLibrary().get_registry()
    matches0 = [item for item in datalib.get_library_overview() if 'DataLoader' in item]
    assert len(matches0), "No DataLoader found"
    logger.debug("Found DataLoader")
    data_handler = datalib.build_object('DataLoader')

    pipelib = PipelineLibrary().get_registry()
    matches1 = [item for item in pipelib.get_library_overview() if settings_data.pipeline in item]
    assert len(matches1), "No Pipeline found"
    logger.debug("Found Pipeline")
    pipe = pipelib.build_object(settings_data.pipeline)
    mode = 'merge' in settings_data.pipeline.lower()

    return settings_data, settings_thr, data_handler, pipe, mode


def select_process_pipeline(object_dataloader, object_pipeline, sets_load_data: SettingsData,
                            sets_load_thread: SettingsThread=RecommendedSettingsThread) -> None:
    """Function for handling the processing pipeline
    :param object_dataloader:   object dataloader
    :param object_pipeline:     object pipeline
    :param sets_load_data:      Dataclass with settings for getting data to analyse
    :param sets_load_thread:    Dataclass with settings for handling the processing threads
    :return:                    None
    """
    define_logger_runtime(False)
    logger = getLogger(__name__)

    # ----- Preparation: Module calling -----
    logger.info("Running framework for end-to-end neural signal processing (DeNSPP)")
    logger.info("Step #1: Loading data")
    logger.info("==================================================================")

    datahand = object_dataloader(sets_load_data)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.build_mapping()

    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    logger.info("Step #2: Processing data")
    logger.info("==================================================================")
    thr_station = ProcessingData(
        pipeline=object_pipeline,
        settings=sets_load_thread,
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


def select_process_merge(object_dataloader, object_pipeline) -> None:
    """Function for preparing and starting the merge process for generating datasets
    :param object_dataloader:   DataLoader object
    :param object_pipeline:     Pipeline object
    """
    logger = getLogger(__name__)
    logger.info("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings = YamlHandler(
        template=DefaultSettingsData,
        path='config',
        file_name='Config_Merge'
    ).get_class(SettingsData)

    # ---- Merging spike frames from several files to one file
    merge_handler = MergeDatasets(object_pipeline, settings, True)
    merge_handler.get_frames_from_dataset(
        data_loader=object_dataloader,
        cluster_class_avai=False,
        process_points=list()
    )
    merge_handler.merge_data_from_diff_files()
    merge_handler.save_merged_data_in_npyfile()

    # --- Merging the frames to new cluster device
    logger.info("=========================================================")
    logger.info("Final Step with merging cluster have to be done in MATLAB")

