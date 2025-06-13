from denspp.offline.pipeline.pipeline_handler import start_processing_pipeline, select_process_pipeline, select_process_merge


if __name__ == "__main__":
    settings_data, settings_thr, data_handler, pipe, mode_merge = start_processing_pipeline()
    if mode_merge:
        select_process_merge(
            object_dataloader=data_handler,
            object_pipeline=pipe
        )
    else:
        select_process_pipeline(
            object_dataloader=data_handler,
            object_pipeline=pipe,
            sets_load_data=settings_data,
            sets_load_thread=settings_thr
        )
