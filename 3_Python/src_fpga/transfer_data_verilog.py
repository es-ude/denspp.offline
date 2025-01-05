import os
from scipy.io import loadmat

from src_neuro.call_spike import DataLoader
from package.data_call.call_handler import SettingsDATA
from src_neuro.pipeline_data import Pipeline
import src_fpga.translate_frames_verilog as verilog_frame
import src_fpga.translate_timeseries_verilog as verilog_time
import src_fpga.translate_timeseries_veriloga as veriloga_time


settings_data = SettingsDATA(
    # path='../2_Data',
    # path='/media/erbsloeh/ExtremeSSD/0_Invasive',
    path='C:/HomeOffice/Data_Neurosignal',
    data_set=1, data_case=0, data_point=0,
    t_range=[0.93, 1.63],
    ch_sel=[],
    fs_resample=50e3
)


def do_data_transfer_timeseries_vivado(path: str, bitsize: int) -> None:
    """Routine for transfering neural rawdata for Vivado simulations
    (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the Settings
    afe = Pipeline(settings_data.fs_resample)

    # ------ Loading Data: Getting the data
    print("... loading the datasets")
    datahandler = DataLoader(settings_data)
    datahandler.do_call()
    datahandler.do_cut()
    datahandler.do_resample()
    data = datahandler.get_data()

    xpos = datahandler.generate_xpos_label(0)
    afe.run_minimal(data.data_raw[0])

    # --- Transfer to verilog files
    if not os.path.exists(path):
        os.mkdir(path)
    path2save = os.path.join(path, 'fpga_sim')

    verilog_time.create_testbench(
        path2save=path2save,
        output_bitsize=bitsize,
        use_trigger=True,
        fs=settings_data.fs_resample
    )
    verilog_time.translate_data_memory(
        raw_data=afe.signals.x_adc,
        trigger=xpos,
        path2save=path2save,
        output_bitsize=bitsize,
        fs=settings_data.fs_resample
    )


def do_data_transfer_timeseries_cadence(path: str) -> None:
    """Routine for transfering neural rawdata for Vivado simulations
        (mode = 'Vivado' or 'Cadence')"""
    print('\nTransfering time series signal (raw data) from neural datasets into Vivado testbenches')
    # --- Loading the src_neuro

    # ------ Loading Data: Getting the data
    print("... loading the datasets")
    datahandler = DataLoader(settings_data)
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


def do_read_frames(path2read_mat: str, path2save: str, bitsize_frame: int) -> None:
    """Processing pipeline for translating data sources into Vivado simulator suitable files"""
    print("Do read frames from")

    # --- Reading data
    data_frames = loadmat(path2read_mat)
    frames_in = data_frames['frames_in']
    frames_cl = data_frames['frames_cluster']

    verilog_frame.create_testbench(
        bitsize_frame=bitsize_frame, num_frames=frames_in.shape[0], size_frame=frames_in.shape[1],
        use_trigger=True, max_value_trigger=int(max(frames_cl)),
        path2save=path2save, file_name='frame_tb')
    verilog_frame.translate_data_memory(
        frame_in=frames_in, bitsize_frame=bitsize_frame, cluster=frames_cl,
        path2save=path2save, file_name='frame_call'
    )


if __name__ == "__main__":
    do_data_transfer_timeseries_vivado('../../data')
    # do_data_transfer_timeseries_cadence('data')
