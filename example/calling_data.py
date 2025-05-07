from denspp.offline.data_call.call_handler import DefaultSettingsData
from denspp.offline.pipeline.plot_mea import plot_mea_transient_total
from src_pipe.call_data import DataLoader


if __name__ == "__main__":
    settings = DefaultSettingsData
    settings.data_set = 'mcs_fzj'
    settings.data_case = 0
    settings.fs_resample = 20e3

    # --- Pipeline
    data_loader = DataLoader(settings)
    data_loader.do_call()
    data_loader.do_cut()
    # data_loader.do_resample()
    data_loader.do_mapping()
    data = data_loader.get_data()

    plot_mea_transient_total(data.data_raw, data, '../../runs/test', do_global_limit=True)
    plot_mea_transient_total(data.data_raw, data, '../../runs/test', do_global_limit=False)
    print(data)
