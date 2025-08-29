from denspp.offline.pipeline.pipeline_handler import start_processing_pipeline, select_process_pipeline, select_process_merge


if __name__ == "__main__":
    settings_data, settings_thr, data_handler, pipe = start_processing_pipeline()
    if settings_data.do_merge:
        select_process_merge(
            object_dataloader=data_handler,
            object_pipeline=pipe,
            sets_load_data=settings_data,
        )
    else:
        select_process_pipeline(
            object_dataloader=data_handler,
            object_pipeline=pipe,
            sets_load_data=settings_data
        )
