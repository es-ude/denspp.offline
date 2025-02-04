from denspp.offline.pipeline.pipeline_handler import start_pipeline_processing


if __name__ == "__main__":
    from src_neuro.call_spike import DataLoader
    from src_neuro.pipeline_v1 import Pipeline

    start_pipeline_processing(DataLoader, Pipeline)
