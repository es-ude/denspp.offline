from denspp.offline.logger import define_logger_runtime
from denspp.offline.pipeline.pipeline_handler import start_pipeline_processing


if __name__ == "__main__":
    from src_pipe.call_data import DataLoader
    from src_pipe.pipeline_v0 import Pipeline

    define_logger_runtime(False)
    start_pipeline_processing(DataLoader, Pipeline)
