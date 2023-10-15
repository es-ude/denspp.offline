import numpy as np
from package.data_call import DataController
from pipeline.pipeline_data import Settings, Pipeline
from package.fpga.translation_timeseries_1ch import translate_data_veriloga, creating_testbench_verilog, translate_data_verilog_memory
from package.fpga.translation_weights_dnn import read_model_weights


def do_data_transfer(mode='Vivado') -> None:
    """Routine for transfering neural rawdata for Vivado simulations
    (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the pipeline
    afe_set = Settings()
    afe_set.SettingsDATA.t_range = [10, 12]
    afe = Pipeline(afe_set)
    # ------ Loading Data: Getting the data
    print("... loading the datasets")
    datahandler = DataController(afe_set.SettingsDATA)
    datahandler.do_call()
    datahandler.do_cut()
    datahandler.do_resample()
    data = datahandler.get_data()
    afe.run_input(data.data_raw[0])

    if mode == 'Vivado':
        u_in = afe.x_adc
    elif mode == 'Cadence':
        u_in = afe.u_in
    else:
        u_in = np.zeros(shape=(100, ), dtype=int)

    # --- Transfer to verilog files
    if mode == 'Vivado':
        path2save = 'data/fpga_sim/neural_data'
        bit_size = 12
        sampling_rate = afe_set.SettingsADC.fs_adc
        creating_testbench_verilog(
            path2save=path2save,
            output_bitsize=bit_size,
            use_trigger=False,
            fs=sampling_rate
        )
        translate_data_verilog_memory(
            raw_data=u_in,
            path2save=path2save,
            output_bitsize=bit_size,
            fs=sampling_rate
        )
    elif mode == 'Cadence':
        path2save = 'data/asic_sim'
        translate_data_veriloga(
            raw_data=u_in,
            path2save=path2save,
            username='erbsloeh'
        )


def do_read_dnn_weights() -> None:
    """Routine for reading DNN weights from trained model"""
    read_model_weights()


if __name__ == "__main__":
    do_data_transfer('Cadence')
