from denspp.offline.pipeline.pipeline_handler import start_pipeline_processing


if __name__ == "__main__":
    from denspp.offline.template.call_template import DataLoader
    from denspp.offline.template.pipeline_v0 import Pipeline

    start_pipeline_processing(DataLoader, Pipeline)
