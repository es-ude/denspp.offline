from logging import getLogger
from denspp.offline.logger import define_logger_runtime
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.data_call import SettingsData, DefaultSettingsData
from denspp.offline.data_call.merge_datasets_frames import MergeDatasets
from denspp.offline.pipeline.pipeline_cmds import DataloaderLibrary, PipelineLibrary
from denspp.offline.pipeline.multithread import MultithreadHandler


def start_processing_pipeline(sets_load_data: SettingsData=DefaultSettingsData):
    """Function for preparing the pipeline preprocessing
    :param sets_load_data:      Dataclass with settings for getting data to analyse
    :return:                    Tuple with (0) Settings class for data, (1) Settings class for thread handling, (2) DataLoader, (3) Pipeline
    """
    # --- Getting the YAML files
    logger = getLogger(__name__)
    if sets_load_data == DefaultSettingsData:
        settings_data = YamlHandler(
            template=sets_load_data,
            path='config',
            file_name='Config_PipelineData'
        ).get_class(SettingsData)
    else:
        settings_data = sets_load_data
    logger.debug("Load YAML configs")

    # --- Getting the Pipeline and DataLoader
    datalib = DataloaderLibrary().get_registry()
    matches0 = [item for item in datalib.get_library_overview() if 'DataLoader' in item]
    assert len(matches0), "No DataLoader found"
    logger.debug("Found DataLoader")
    data_handler = datalib.build_object('DataLoader')

    pipelib = PipelineLibrary().get_registry()
    search_name = sets_load_data.pipeline + ('_Merge' if sets_load_data.do_merge else '')
    matches1 = [item for item in pipelib.get_library_overview() if search_name in item]
    assert len(matches1), "No Pipeline found"
    logger.debug("Found Pipeline")
    pipe = pipelib.build_object(matches1[0])

    return settings_data, data_handler, pipe


def select_process_pipeline(object_dataloader, object_pipeline, sets_load_data: SettingsData) -> None:
    """Function for handling the processing pipeline
    :param object_dataloader:   object dataloader
    :param object_pipeline:     object pipeline
    :param sets_load_data:      Dataclass with settings for getting data to analyse
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
    thr_station = MultithreadHandler(
        num_workers=1
    )
    dut = object_pipeline(dataIn.fs_used)
    thr_station.do_processing(
        data=dataIn.data_raw,
        chnnl_id=dataIn.electrode_id,
        func=dut.run
    )

    # --- Plot all plots and save results
    logger.info("Step #3: Saving results and plotting")
    logger.info("==================================================================")
    thr_station.do_save_results(path2save=dut.path2save)


def select_process_merge(object_dataloader, object_pipeline, sets_load_data: SettingsData) -> None:
    """Function for preparing and starting the merge process for generating datasets
    :param object_dataloader:   DataLoader object
    :param object_pipeline:     Pipeline object
    :param sets_load_data:      SettingsData object
    :return:                    None
    """
    logger = getLogger(__name__)
    logger.info("Running framework for end-to-end neural signal processing (DeNSPP)")
    logger.info("Building datasets from transient data")

    # ---- Merging spike frames from several files to one file
    merge_handler = MergeDatasets(object_pipeline, object_dataloader, sets_load_data, True)
    merge_handler.get_frames_from_dataset(
        concatenate_id=False,
        process_points=list()
    )
    merge_handler.merge_data_from_diff_files()
    merge_handler.save_merged_data_in_npyfile()

    # --- Merging the frames to new cluster device
    logger.info("=========================================================")
    logger.info("Final Step with merging cluster have to be done in MATLAB")

