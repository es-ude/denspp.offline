from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
import torch
from fxpmath import Fxp


def read_model_weights() -> None:
    """Reading DNN model weights for usage in FPGAs"""
    path2model = "runs/20230531_164911_train_dnn_dae_v1/model_369"
    # path2model = "runs/20230830_162608_train_dnn_ae_v1/model_474"

    model = torch.load(path2model)

    print("Output of weights in AI model")

    ite = 0
    for name, param in model.named_parameters():
        print("--------------------")
        print(f"\nIteration i={ite}")
        print(name)
        A = param
        print(param)

    print("TEST 1")


# TODO: Write the code for verilogA testings
def translate_data_veriloga(raw_data: np.ndarray, output_bitsize: int,
                            path2save='data/fpga_sim', file_name='test_mem',
                            trigger=np.zeros(shape=(1,), dtype=int),
                            fs=20e3, analog=False
                            ) -> None:
    """Translating raw_data and trigger signals (opt.) from Python to VerilogA module for Cadence simulations"""
    trigger_avaible = True if trigger.size != 1 else False
    version = 'v1.0'

    if path2save != '' and isdir(path2save) == False:
        mkdir(path2save)

    path2mem = join(path2save, 'data_raw.mem')
    path2trg = join(path2save, 'data_trg.mem')

    size_input = int(raw_data.size)
    size_cnt = int(np.ceil(np.log2(size_input)))

    raw_data0 = np.array(raw_data, dtype=int)
    trgg0 = np.array(trigger, dtype=int)


def template_get_testbench(path2save='data/fpga_sim', output_bitsize=12) -> None:
    print('\nTemplate for creating files (*.v, *.mem) for time series signals in testbenches')
    # Settings for sinusoidal waveform
    amp = np.power(2, output_bitsize-1) - 2
    off = 0
    f0 = 2
    # Signal generation
    time = np.linspace(0, 1, 5000)
    wfg = np.round(off + amp * 0.5 * np.sin(2 * np.pi * time * f0))
    trgg = 0 * wfg
    trgg[wfg > off] = 1
    trgg[wfg <= off] = 0

    # Running code
    creating_testbench_verilog(path2save=path2save)
    translate_data_verilog_memory(wfg, output_bitsize, trigger=trgg, path2save=path2save)


# TODO: Building routine for running simulation until sim_end_flag rises
def creating_testbench_verilog(path2save='data/fpga_sim', file_name='TB_NeuralSim',
                               output_bitsize=12, use_trigger=False,
                               fs=20e3) -> None:
    """Creating the testbench environment in Verilog for using in digital design software"""
    period_time = int(0.5* 1e9/fs)
    size_period = int(np.ceil(np.log2(period_time)))
    if path2save != '' and isdir(path2save) == False:
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
        tb_handler.write(f'\tlocalparam CLK_CYC = {size_period}\'d{period_time};\n\n')
        tb_handler.write(f'\treg clk, nrst, en_sim;\n')
        tb_handler.write(f'\twire signed[{output_bitsize-1}:0] data_out0;\n')
        if use_trigger:
            tb_handler.write(f'\twire trgg_out0;\n')
        tb_handler.write(f'\twire sim_done0;\n\n')
        tb_handler.write(f'\t// Choice if files with loading from memory (*_Mem) or from module (*_Mod)\n')
        tb_handler.write(f'\tNeuralSim_Mem DUT0(\n')
        tb_handler.write(f'\t\t.CLK_ADC(clk),\n')
        tb_handler.write(f'\t\t.nRST(nrst),\n')
        tb_handler.write(f'\t\t.EN(en_sim),\n')
        tb_handler.write(f'\t\t.DATA_OUT(data_out0),\n')
        if use_trigger:
            tb_handler.write(f'\t\t.TRGG_OUT(trgg_out0),\n')
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
        tb_handler.write(f'\t\ten_sim = 1\'b1;\n')
        # tb_handler.write(f'\t\t#(100_000) $stop;\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'endmodule')


def translate_data_verilog_memory(raw_data: np.ndarray, output_bitsize: int,
                            path2save='data\\fpga_sim', file_name='test_mem',
                            trigger=np.zeros(shape=(1,), dtype=int),
                            fs=20e3, output_signed=True
                            ) -> None:
    """Translating raw_data and trigger signals (opt.) from Python to Verilog module for testbenches"""
    trigger_avaible = True if trigger.size != 1 else False
    version = 'v1.0'

    if path2save != '' and isdir(path2save) == False:
        mkdir(path2save)
    path2mem = join(path2save, 'data_raw.mem')
    path2trg = join(path2save, 'data_trg.mem')

    size_input = int(raw_data.size)
    size_cnt = int(np.ceil(np.log2(size_input)))

    raw_data0 = np.array(raw_data, dtype=int)
    trgg0 = np.array(trigger, dtype=int)

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
        v_handler.write(f'// Size of input array: {size_input}\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        # --- Header 2: Example for using in testbench
        v_handler.write(f'// Code Example for Implementation in Testbench:\n//\n')
        v_handler.write(f'//\tNeuralSim_Mem DataSim(\n')
        v_handler.write(f'//\t\t.CLK_ADC(clk0),\n')
        v_handler.write(f'//\t\t.nRST(reset_n),\n')
        v_handler.write(f'//\t\t.EN(enable_sim)\n')
        v_handler.write(f'//\t\t.DATA_OUT(data_sim),\n')
        if trigger_avaible:
            v_handler.write(f'//\t\t.TRGG_OUT(trgg_sim),\n')
        v_handler.write(f'//\t\t.DATA_END(sim_end)\n')
        v_handler.write(f'//\t);\n')
        v_handler.write(f'// ---------------------------------------------------\n\n')
        # --- Code: Module initialization
        v_handler.write(f'module NeuralSim_Mem(\n')
        v_handler.write(f'\tinput wire CLK_ADC,\n')
        v_handler.write(f'\tinput wire nRST,\n')
        v_handler.write(f'\tinput wire EN,\n')
        v_handler.write(f'\toutput wire signed [{output_bitsize - 1}:0] DATA_OUT,\n')
        if trigger_avaible:
            v_handler.write(f'\toutput wire TRGG_OUT,\n')
        v_handler.write(f'\toutput wire DATA_END\n')
        v_handler.write(f');\n\n')
        v_handler.write(f'reg [{size_cnt - 1}:0] cnt_pos;\n\n')
        if trigger_avaible:
            v_handler.write(f'reg bram_trgg [0:{size_input - 1}];\n')
            v_handler.write(f'assign TRGG_OUT = (EN && !DATA_END) ? bram_trgg[cnt_pos] : 1\'d0;\n')
        v_handler.write(f'reg signed [{output_bitsize - 1}:0] bram_data [0:{size_input - 1}];\n')
        v_handler.write(f'assign DATA_OUT = (EN && !DATA_END) ? bram_data[cnt_pos] : {output_bitsize}\'d0;\n\n')
        v_handler.write(f'assign DATA_END = (cnt_pos == \'d{size_input - 1});\n\n')
        v_handler.write(f'initial begin\n')
        v_handler.write(f'\t$readmemh("data_raw.mem", bram_data);\n')
        if trigger_avaible:
            v_handler.write(f'\t$readmemb("data_trg.mem", bram_trgg);\n')
        v_handler.write(f'end\n\n')
        v_handler.write(f'always@(posedge CLK_ADC or negedge nRST) begin\n')
        v_handler.write(f'\tif(!nRST) begin\n')
        v_handler.write(f'\t\tcnt_pos = {size_cnt}\'d0;\n')
        v_handler.write(f'\tend else begin\n')
        v_handler.write(f'\t\tcnt_pos = cnt_pos + ((EN && !DATA_END) ? {size_cnt}\'d1 : {size_cnt}\'d0);\n')
        v_handler.write(f'\tend\n')
        v_handler.write(f'end\n\n')
        v_handler.write(f'endmodule\n')

    # --- Second file: Memory of rawdata ($readmemh)
    print('... transfer rawdata to memory file (*.mem)')
    with open(path2mem, 'w') as mem_handler:
        for idx, val in enumerate(raw_data0):
            trans_val = Fxp(val, signed=output_signed, n_word=output_bitsize)
            mem_handler.write(f'{trans_val.hex(True)[2:]}\n')

    # --- Third file: Memory of trigger ($readmemb)
    if trigger_avaible:
        print('... transfer trigger signals to memory file (*.mem)')
        with open(path2trg, 'w') as mem_handler:
            for idx, val in enumerate(trgg0):
                trans_val = Fxp(val, signed=False, n_word=1)
                mem_handler.write(f'{trans_val.bin(False)}\n')


def translate_data_verilog_module(raw_data: np.ndarray, output_bitsize: int,
                            path2save='data\\fpga_sim', file_name='test_mod',
                            trigger=np.zeros(shape=(1,), dtype=int),
                            fs=20e3) -> None:
    """Translating raw_data and trigger signals (opt.) from Python to Verilog module for testbenches"""
    trigger_avaible = True if trigger.size != 1 else False
    version = 'v1.0'
    if path2save != '' and isdir(path2save) == False:
        mkdir(path2save)

    print('... transfer raw_data to verilog file (*.v)')
    size_input = int(raw_data.size)
    size_cnt = int(np.ceil(np.log2(size_input)))

    raw_data0 = np.array(raw_data, dtype=int)
    trgg0 = np.array(trigger, dtype=int)
    print('... create verilog file for handling data (*.v)')
    with open(join(path2save, file_name + '.v'), 'w') as v_handler:
        # --- Header 1: Used tools and data
        v_handler.write(f'// ---------------------------------------------------\n')
        v_handler.write(f'// Raw data converting from DeNSSP to Verilog ({version})\n')
        v_handler.write(f'// Testbench file for playing data sets in simulations\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        v_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        v_handler.write(f'// Original sampling rate fs = {fs/1e3: .2f} kHz\n')
        v_handler.write(f'// Size of input array: {size_input}\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        # --- Header 2: Example for using in testbench
        v_handler.write(f'// Code Example for Implementation in Testbench:\n//\n')
        v_handler.write(f'//\tNeuralSim_Mod DataSim(\n')
        v_handler.write(f'//\t\t.CLK_ADC(clk0),\n')
        v_handler.write(f'//\t\t.nRST(reset_n),\n')
        v_handler.write(f'//\t\t.EN(enable_sim)\n')
        v_handler.write(f'//\t\t.DATA_OUT(data_sim),\n')
        if trigger_avaible:
            v_handler.write(f'//\t\t.TRGG_OUT(trgg_sim),\n')
        v_handler.write(f'//\t\t.DATA_END(sim_end)\n')
        v_handler.write(f'//\t);\n')
        v_handler.write(f'// ---------------------------------------------------\n\n')
        # --- Code: Module initialization
        v_handler.write(f'module NeuralSim_Mod(\n')
        v_handler.write(f'\tinput wire CLK_ADC,\n')
        v_handler.write(f'\tinput wire nRST,\n')
        v_handler.write(f'\tinput wire EN,\n')
        v_handler.write(f'\toutput wire signed [{output_bitsize-1}:0] DATA_OUT,\n')
        if trigger_avaible:
            v_handler.write(f'\toutput wire TRGG_OUT,\n')
        v_handler.write(f'\toutput wire DATA_END\n')
        v_handler.write(f');\n\n')
        v_handler.write(f'reg [{size_cnt-1}:0] cnt_pos;\n')
        v_handler.write(f'wire signed [{output_bitsize-1}:0] bram_data [{size_input-1}:0];\n')
        if trigger_avaible:
            v_handler.write(f'wire bram_trgg [{size_input-1}:0];\n')
            v_handler.write(f'assign TRGG_OUT = bram_trgg[cnt_pos];\n')
        v_handler.write(f'\n')
        v_handler.write(f'assign DATA_OUT = (EN && !DATA_END) ? bram_data[cnt_pos] : {output_bitsize}\'d0;\n')
        v_handler.write(f'assign DATA_END = (cnt_pos == \'d{size_input-1});\n\n')
        v_handler.write(f'always@(posedge CLK_ADC or negedge nRST) begin\n')
        v_handler.write(f'\tif(!nRST) begin\n')
        v_handler.write(f'\t\tcnt_pos = {size_cnt}\'d0;\n')
        v_handler.write(f'\tend else begin\n')
        v_handler.write(f'\t\tcnt_pos = cnt_pos + ((EN && !DATA_END) ? {size_cnt}\'d1 : {size_cnt}\'d0);\n')
        v_handler.write(f'\tend\n')
        v_handler.write(f'end\n\n')
        v_handler.write(f'// --- Data save in BRAM\n')
        # --- Code: Data values
        for idx, val in enumerate(raw_data0):
            if val >= 0:
                out_string = f'{output_bitsize}\'d{val};\n'
            else:
                out_string = f'-{output_bitsize}\'d{np.abs(val)};\n'

            v_handler.write(f'\tassign bram_data[{idx}] = ' + out_string)
        # --- Code: Trigger values
        if trigger_avaible:
            v_handler.write(f'\n')
            for idx, val in enumerate(trgg0):
                v_handler.write(f'\tassign bram_trgg[{idx}] = 1\'d{val};\n')
        # --- End of module
        v_handler.write(f'\nendmodule\n')
