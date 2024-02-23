import os
from scipy.io import loadmat

from package.data_call.data_call_common import DataController
from src_data.pipeline_data import Settings, Pipeline
import package.fpga.verilog_translate_frames as verilog_frame
import package.fpga.verilog_translate_timeseries_1ch as verilog_time
import package.fpga.veriloga_translate_timeseries_1ch as veriloga_time
from package.fpga.verilog_translate_weights import read_model_weights


def do_data_transfer_timeseries_vivado(path: str) -> None:
    """Routine for transfering neural rawdata for Vivado simulations
    (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the src_neuro
    afe_set = Settings()
    afe_set.SettingsDATA.t_range = [0.93, 1.63]
    afe = Pipeline(afe_set)
    # ------ Loading Data: Getting the data
    print("... loading the datasets")
    datahandler = DataController(afe_set.SettingsDATA)
    datahandler.do_call()
    datahandler.do_cut()
    datahandler.do_resample()
    data = datahandler.get_data()
    afe.run_minimal(data.data_raw[0])

    trgg = 0

    # --- Transfer to verilog files
    if not os.path.exists(path):
        os.mkdir(path)
    path2save = os.path.join(path, 'fpga_sim')

    bit_size = 12
    sampling_rate = afe_set.SettingsADC.fs_adc
    verilog_time.create_testbench(
        path2save=path2save,
        output_bitsize=bit_size,
        use_trigger=True,
        fs=sampling_rate
    )
    verilog_time.translate_data_memory(
        raw_data=afe.signals.x_adc,
        trigger= (data.spike_xpos[0] - data.spike_offset_us[0] * data.data_fs_used * 1e6) / data.data_fs_used * afe.signals.fs_dig,
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
    # ------ Loading Data: Getting the data
    print("... loading the datasets")
    datahandler = DataController(afe_set.SettingsDATA)
    datahandler.do_call()
    datahandler.do_cut()
    datahandler.do_resample()
    data = datahandler.get_data()
    u_in = data.data_raw[0]

    if not os.path.exists(path):
        os.mkdir(path)
    path2save = os.path.join(path, 'asic_sim')

    veriloga_time.translate_data(
        raw_data=u_in,
        path2save=path2save,
        username='erbsloeh'
    )


def do_read_frames(path: str) -> None:
    print("Do read frames from")

    # --- Reading data
    data_frames = loadmat('data/2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat')
    frames_in = data_frames['frames_in']
    frames_cl = data_frames['frames_cluster']

    bitsize_frame = 12

    verilog_frame.create_testbench(
        bitsize_frame=bitsize_frame, num_frames=frames_in.shape[0], size_frame=frames_in.shape[1],
        use_trigger=True, max_value_trigger=int(max(frames_cl)),
        path2save=path, file_name='frame_tb')
    verilog_frame.translate_data_memory(
        frame_in=frames_in, bitsize_frame=bitsize_frame, cluster=frames_cl,
        path2save=path, file_name='frame_call'
    )


def do_read_dnn_weights() -> None:
    """Routine for reading DNN weights from trained model"""
    path2model = "runs/20230531_164911_train_dnn_dae_v1/model_369"
    # path2model = "runs/20230830_162608_train_dnn_ae_v1/model_474"

    read_model_weights(path2model)


if __name__ == "__main__":
    do_data_transfer_timeseries_vivado('data')
    # do_data_transfer_timeseries_cadence('data')
    # do_read_frames('data')
    # do_read_dnn_weights()
