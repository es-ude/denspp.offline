import os
from scipy.io import loadmat

from package.data.data_call import DataController
from src_neuro.pipeline_data import Settings, Pipeline
import package.fpga.verilog_translate_timeseries_1ch as verilog_time
import package.fpga.veriloga_translate_timeseries_1ch as veriloga_time
import package.fpga.verilog_translate_frames as verilog_frame
from package.fpga.verilog_translate_weights import read_model_weights


def do_data_transfer_timeseries_vivado(path: str) -> None:
    """Routine for transfering neural rawdata for Vivado simulations
    (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the src_neuro
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

    u_in = afe.x_adc

    # --- Transfer to verilog files
    if not os.path.exists(path):
        os.mkdir(path)
    path2save = os.path.join(path, 'fpga_sim')

    bit_size = 12
    sampling_rate = afe_set.SettingsADC.fs_adc
    verilog_time.create_testbench(
        path2save=path2save,
        output_bitsize=bit_size,
        use_trigger=False,
        fs=sampling_rate
    )
    verilog_time.translate_data_memory(
        raw_data=u_in,
        path2save=path2save,
        output_bitsize=bit_size,
        fs=sampling_rate
    )


def do_data_transfer_timeseries_cadence(path: str) -> None:
    """Routine for transfering neural rawdata for Vivado simulations
        (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the src_neuro
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

    u_in = afe.u_in

    if not os.path.exists(path):
        os.mkdir(path)
    path2save = os.path.join(path, 'asic_sim')

    veriloga_time.translate_data(
        raw_data=u_in,
        path2save=path2save,
        username='erbsloeh'
    )


def do_read_frames() -> None:
    print("Do read frames from")

    # --- Reading data
    data_frames = loadmat('data/2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat')
    frames_in = data_frames['frames_in']
    frames_cl = data_frames['frames_cluster']

    bitsize_frame = 12

    verilog_frame.create_testbench(
        bitsize_frame=bitsize_frame, num_frames=frames_in.shape[0], size_frame=frames_in.shape[1],
        use_trigger=True, max_value_trigger=int(max(frames_cl)),
        path2save='data', file_name='frame_tb')
    verilog_frame.translate_data_memory(
        frame_in=frames_in, bitsize_frame=bitsize_frame, cluster=frames_cl,
        path2save='data', file_name='frame_call'
    )


def do_read_dnn_weights() -> None:
    """Routine for reading DNN weights from trained model"""
    read_model_weights()


if __name__ == "__main__":
    # do_data_transfer_timeseries_vivado('data')
    # do_data_transfer_timeseries_cadence('data')

    do_read_frames()
