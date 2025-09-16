from dataclasses import dataclass
from logging import getLogger
from denspp.offline.logger import define_logger_runtime
from denspp.offline.data_call import SettingsData, DefaultSettingsData, MergeDataset
from denspp.offline.data_format import YamlHandler
from denspp.offline.pipeline import DataloaderLibrary, PipelineLibrary, MultithreadHandler


def _start_processing_pipeline(sets_load_data: SettingsData=DefaultSettingsData):
    """Function for preparing the pipeline preprocessing
    :param sets_load_data:      Dataclass with settings for getting data to analyse
    :return:                    Tuple with (0) Settings class for data, (1) DataLoader, (2) Pipeline
    """
    # --- Getting the YAML files
    logger = getLogger(__name__)
    if sets_load_data == DefaultSettingsData:
        settings_data: SettingsData = YamlHandler(
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


def select_process_merge(object_dataloader, object_pipeline, sets_load_data: SettingsData,
                         frames_xoffset: int=0, list_merging_files: list=(), do_label_concatenation: bool=False) -> None:
    """Function for preparing and starting the merge process for generating datasets
    :param object_dataloader:       DataLoader object
    :param object_pipeline:         Pipeline object
    :param sets_load_data:          SettingsData object
    :param frames_xoffset:          Integer with offset
    :param list_merging_files:      Taking the datapoints of the selected data set to process
    :param do_label_concatenation:  Do concatenation of the class number with increasing id number (useful for non-biological clusters)
    :return:                        None
    """
    logger = getLogger(__name__)
    logger.info("Running framework for end-to-end neural signal processing (DeNSPP)")
    logger.info("Building datasets from transient data")

    # ---- Merging spike frames from several files to one file
    merge_handler = MergeDataset(
        pipeline=object_pipeline,
        dataloader=object_dataloader,
        settings_data=sets_load_data,
        concatenate_id=do_label_concatenation,
    )
    merge_handler.merge_data_from_all_iteration()
    merge_handler.get_frames_from_dataset(
        process_points=list_merging_files,
        xpos_offset=frames_xoffset
    )
    merge_handler.merge_data_from_all_iteration()

    # --- Merging the frames to new cluster device
    logger.info("=========================================================")
    logger.info("Final Step with merging cluster have to be done separately using SortDataset")


@dataclass
class SettingsMerging:
    """Class for defining the properties for merging datasets
    Attributes:
        taking_datapoints:  List with data_points to process [Default: [] -> taking all]
        do_label_concat:    Boolean for concatenating the
        xoffset:            Integer with delayed positions applied on frame/window extraction
    """
    taking_datapoints: list[int]
    do_label_concat: bool
    xoffset: int


DefaultSettingsMerging = SettingsMerging(
    taking_datapoints=[],
    do_label_concat=False,
    xoffset=0,
)


def run_transient_data_processing() -> None:
    """Function for running the offline data analysis of transient use-specific data
    :return:    None
    """
    define_logger_runtime(False)
    settings_data, data_handler, pipe = _start_processing_pipeline()

    if settings_data.do_merge:
        sets_merge: SettingsMerging = YamlHandler(
            template=DefaultSettingsMerging,
            path='config',
            file_name='Config_Merging'
        ).get_class(SettingsMerging)

        select_process_merge(
            object_dataloader=data_handler,
            object_pipeline=pipe,
            sets_load_data=settings_data,
            list_merging_files=sets_merge.taking_datapoints,
            do_label_concatenation=sets_merge.do_label_concat,
            frames_xoffset=sets_merge.xoffset,
        )
    else:
        select_process_pipeline(
            object_dataloader=data_handler,
            object_pipeline=pipe,
            sets_load_data=settings_data
        )
