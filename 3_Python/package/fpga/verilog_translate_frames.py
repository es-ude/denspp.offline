from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
from fxpmath import Fxp


def create_testbench(path2save='data/fpga_sim', file_name='TB_NeuralSim',
                     bitsize_frame=12, size_frame=32, num_frames=6098,
                     max_value_trigger=4, use_trigger=False, fs=20e3) -> None:
    """Creating the testbench environment in Verilog for using in digital design software (frames)"""
    period_time = int(0.5 * 1e9/fs)
    size_period = int(np.ceil(np.log2(period_time)))
    size_cnt_cluster = int(np.ceil(np.log2(max_value_trigger)))
    size_cnt_frame = int(np.ceil(np.log2(size_frame)))
    size_cnt_runs = int(np.ceil(np.log2(num_frames)))
    if path2save != '' and not isdir(path2save):
        mkdir(path2save)

    tb_name = 'Testbench for running dataset simulator'
    print('... create testbench verilog file (*.v)')
    with open(join(path2save, file_name + '.v'), 'w') as tb_handler:
        tb_handler.write(f'`timescale 1ns / 1ps\n')
        tb_handler.write(f'// -----------------------------------------------------------------\n')
        tb_handler.write(f'// Company: UDE-ES\n')
        tb_handler.write(f'// Design Name: {tb_name}\n')
        tb_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        tb_handler.write(f'// Target Devices: Simulation file\n')
        tb_handler.write(f'// -----------------------------------------------------------------\n\n')
        tb_handler.write(f'module {file_name}();\n')
        tb_handler.write(f'\tlocalparam CLK_CYC = {size_period}\'d{period_time};\n')
        tb_handler.write(f'\treg clk, nrst, en_sim;\n')
        tb_handler.write(f'\treg [{size_cnt_runs - 1}:0] cnt_frame;\n')
        tb_handler.write(f'\treg [{size_cnt_frame - 1}:0] cnt_pos;\n\n')
        tb_handler.write(f'\twire new_frame_start;\n')
        tb_handler.write(f'\twire signed [{bitsize_frame - 1}:0] frame_transient;\n')
        tb_handler.write(f'\twire signed [{size_frame * bitsize_frame - 1}:0] frame_out;\n')
        tb_handler.write(f'\twire signed [{bitsize_frame - 1}:0] frame_out_sliced [0:{size_frame - 1}];\n')
        tb_handler.write(f'\tgenvar i0;\n')
        tb_handler.write(f'\tfor(i0 = \'d0; i0 < {size_frame}; i0 = i0 + \'d1) begin\n')
        tb_handler.write(f'\t\tassign frame_out_sliced[i0] = frame_out[i0*{bitsize_frame}+:{bitsize_frame}];\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'\tassign frame_transient = frame_out_sliced[cnt_pos];\n')
        tb_handler.write(f'\tassign new_frame_start = (cnt_pos == {size_cnt_frame}\'d0);\n')
        if use_trigger:
            tb_handler.write(f'\twire [{size_cnt_cluster-1}:0] cluster_out;\n')
        tb_handler.write(f'\twire sim_done0;\n\n')
        tb_handler.write(f'\t// Choice if files with loading from memory\n')
        tb_handler.write(f'\tNeuralSim_Frame_Mem DUT0(\n')
        tb_handler.write(f'\t\t.SEL(cnt_frame),\n')
        tb_handler.write(f'\t\t.EN(en_sim),\n')
        tb_handler.write(f'\t\t.FRAME_OUT(frame_out),\n')
        if use_trigger:
            tb_handler.write(f'\t\t.CLUSTER_OUT(cluster_out),\n')
        tb_handler.write(f'\t\t.DATA_END(sim_done0)\n')
        tb_handler.write(f'\t);\n\n')
        tb_handler.write(f'\t// Control scheme for getting the data dependent on the sampling clock\n')
        tb_handler.write(f'\talways begin\n')
        tb_handler.write(f'\t\t#(CLK_CYC) clk = ~clk;\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'\tinitial begin\n')
        tb_handler.write(f'\t\tnrst = 1\'d0;\n')
        tb_handler.write(f'\t\tclk = 1\'d0;\n')
        tb_handler.write(f'\t\ten_sim = 1\'d0;\n\n')
        tb_handler.write(f'\t\t# (6* CLK_CYC);   nrst <= 1\'b1;\n')
        tb_handler.write(f'\t\t# (6* CLK_CYC);   nrst <= 1\'b0;\n')
        tb_handler.write(f'\t\t# (6* CLK_CYC);   nrst <= 1\'b1;\n')
        tb_handler.write(f'\t\t# (1* CLK_CYC);   en_sim = 1\'b1;\n')
        tb_handler.write(f'\t\t// #(100_000) $stop;\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'\treg first_start;\n')
        tb_handler.write(f'\talways@(posedge clk or negedge nrst) begin\n')
        tb_handler.write(f'\t\tif(!nrst) begin\n')
        tb_handler.write(f'\t\t\tcnt_frame = {size_cnt_runs}\'d0;\n')
        tb_handler.write(f'\t\t\tcnt_pos = {size_cnt_frame}\'d0;\n')
        tb_handler.write(f'\t\t\tfirst_start = 1\'d0;\n')
        tb_handler.write(f'\t\tend else begin\n')
        tb_handler.write(f'\t\t\tif(cnt_pos == {size_cnt_frame}\'d{size_frame-1}) begin\n')
        tb_handler.write(f'\t\t\t\tcnt_frame = cnt_frame + ((first_start && en_sim && !sim_done0) ? {size_cnt_runs}\'d1 : {size_cnt_runs}\'d0);\n')
        tb_handler.write(f'\t\t\t\tcnt_pos = {size_cnt_frame}\'d0;\n')
        tb_handler.write(f'\t\t\tend else begin\n')
        tb_handler.write(f'\t\t\t\tcnt_frame = cnt_frame;\n')
        tb_handler.write(f'\t\t\t\tcnt_pos = (first_start && en_sim && !sim_done0) ? cnt_pos + {size_cnt_frame}\'d1 : {size_cnt_frame}\'d0;\n')
        tb_handler.write(f'\t\t\tend\n')
        tb_handler.write(f'\t\t\tfirst_start = en_sim;\n')
        tb_handler.write(f'\t\tend\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'endmodule')


def translate_data_memory(frame_in: np.ndarray, bitsize_frame: int,
                          cluster=np.zeros(shape=(1,), dtype=int),
                          path2save='data\\fpga_sim', file_name='test_mem',
                          fs=20e3, output_signed=True) -> None:
    """Translating raw_data and trigger signals (opt.) from Python to Verilog module for testbenches"""
    trigger_avaible = True if cluster.size != 1 else False
    version = 'v1.0'

    if path2save != '' and not isdir(path2save):
        mkdir(path2save)
    mem_name = 'data_frame.mem'
    trg_name = 'data_cluster.mem'
    path2mem = join(path2save, mem_name)
    path2trg = join(path2save, trg_name)

    num_frames = int(frame_in.shape[0])
    size_frame = int(frame_in.shape[1])
    size_cnt = int(np.ceil(np.log2(num_frames)))
    size_cnt_cluster = int(np.ceil(np.log2(max(cluster))))

    frame_in0 = np.array(frame_in, dtype=int)
    cluster0 = np.array(cluster, dtype=int)

    # --- First file for testbench file
    print('... create verilog file for handling data (*.v)')
    with open(join(path2save, file_name + '.v'), 'w') as v_handler:
        # --- Header 1: Used tools and data
        v_handler.write(f'// ---------------------------------------------------\n')
        v_handler.write(f'// Raw data converting from DeNSSP to Verilog ({version})\n')
        v_handler.write(f'// Testbench file for playing data sets in simulations\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        v_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        v_handler.write(f'// Original sampling rate fs = {fs / 1e3: .2f} kHz\n')
        v_handler.write(f'// Num of frames: {num_frames} - [{size_frame-1}:0][{bitsize_frame-1}:0]\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        # --- Header 2: Example for using in testbench
        v_handler.write(f'// Code Example for Implementation in Testbench:\n//\n')
        v_handler.write(f'//\tNeuralSim_Frame_Mem FrameSim(\n')
        v_handler.write(f'//\t\t.SEL(sel0),\n')
        v_handler.write(f'//\t\t.EN(enable_sim)\n')
        v_handler.write(f'//\t\t.FRAME_OUT(frame_sim),\n')
        if trigger_avaible:
            v_handler.write(f'//\t\t.CLUSTER_OUT(cluster_sim),\n')
        v_handler.write(f'//\t\t.DATA_END(sim_end)\n')
        v_handler.write(f'//\t);\n')
        v_handler.write(f'// ---------------------------------------------------\n\n')
        # --- Code: Module initialization
        v_handler.write(f'module NeuralSim_Frame_Mem(\n')
        v_handler.write(f'\tinput wire [{size_cnt - 1}:0] SEL,\n')
        v_handler.write(f'\tinput wire EN,\n')
        v_handler.write(f'\toutput wire signed [{size_frame * bitsize_frame - 1}:0] FRAME_OUT,\n')
        if trigger_avaible:
            v_handler.write(f'\toutput wire [{size_cnt_cluster - 1}:0] CLUSTER_OUT,\n')
        v_handler.write(f'\toutput wire DATA_END\n')
        v_handler.write(f');\n\n')
        if trigger_avaible:
            v_handler.write(f'reg [{size_cnt_cluster-1}:0] bram_trgg [0:{num_frames-1}];\n')
            v_handler.write(f'assign CLUSTER_OUT = (EN && !DATA_END) ? bram_trgg[SEL] : {size_cnt_cluster}\'d0;\n\n')
        v_handler.write(f'reg signed [{bitsize_frame-1}:0] bram_data [0:{num_frames * size_frame - 1}];\n')
        v_handler.write(f'genvar i0;\n')
        v_handler.write(f'for(i0 = \'d0; i0 < {size_frame}; i0 = i0 + \'d1) begin\n')
        v_handler.write(f'\tassign FRAME_OUT[i0*{bitsize_frame}+:{bitsize_frame}] = (EN && !DATA_END) ? '
                        f'bram_data[i0 + SEL * {size_frame}] : {bitsize_frame}\'d0;\n')
        v_handler.write(f'end\n')
        v_handler.write(f'assign DATA_END = (SEL == \'d{num_frames - 1});\n\n')
        v_handler.write(f'initial begin\n')
        v_handler.write(f'\t$readmemh("{mem_name}", bram_data);\n')
        if trigger_avaible:
            v_handler.write(f'\t$readmemb("{trg_name}", bram_trgg);\n')
        v_handler.write(f'end\n\n')
        v_handler.write(f'endmodule\n')

    # --- Second file: Memory of rawdata ($readmemh)
    print('... transfer rawdata to memory file (*.mem)')
    with open(path2mem, 'w') as mem_handler:
        for frame in frame_in0:
            frame_fxp = Fxp(frame, signed=output_signed, n_word=bitsize_frame)
            for val in frame_fxp:
                mem_handler.write(f'{val.hex(True)[2:]}\n')

    # --- Third file: Memory of trigger ($readmemb)
    if trigger_avaible:
        print('... transfer trigger signals to memory file (*.mem)')
        with open(path2trg, 'w') as mem_handler:
            for val in cluster0:
                trans_val = Fxp(val[0], signed=False, n_word=size_cnt_cluster)
                mem_handler.write(f'{trans_val.bin(False)}\n')
