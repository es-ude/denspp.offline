from denspp.offline.pipeline.pipeline_handler import start_pipeline_processing


if __name__ == "__main__":
    from src_pipe.call_data import DataLoader
    from src_pipe.pipeline_v0 import Pipeline

    start_pipeline_processing(DataLoader, Pipeline)
