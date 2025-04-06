from denspp.offline.data_merge.merge_process import start_merge_process


if __name__ == "__main__":
    from denspp.offline.data_merge.pipeline_data import Pipeline
    from src_neuro.call_data import DataLoader

    start_merge_process(DataLoader, Pipeline)
