from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np


# TODO: Write the code for verilogA testings
# TODO: Check if delay and smoothing via VerilogA commands can be included
def translate_data(raw_data: np.ndarray,
                   path2save='data/asic_sim', username='<user>',
                   trigger=np.zeros(shape=(1,), dtype=int),
                   fs=20e3) -> None:
    """Translating raw_data and trigger signals (opt.) from Python to VerilogA module for Cadence simulations"""
    trigger_avaible = True if trigger.size != 1 else False
    version = 'v1.0'

    if path2save != '' and isdir(path2save) == False:
        mkdir(path2save)

    path2mem = join(path2save, 'data_raw.txt')
    path2trg = join(path2save, 'data_trg.txt')

    size_input = int(raw_data.size)
    size_cnt = int(np.ceil(np.log2(size_input)))

    raw_data0 = np.array(raw_data, dtype=float)
    trgg0 = np.array(trigger, dtype=int)

    # --- First file for testbench file
    print('... create testbench verilog file (*.va)')
    with open(join(path2save, 'veriloga.va'), 'w') as tb_handler:
        tb_handler.write('`include "constants.vams"\n')
        tb_handler.write('`include "disciplines.vams"\n\n')
        tb_handler.write(f'// -----------------------------------------------------------------\n')
        tb_handler.write(f'// Company: UDE-ES\n')
        tb_handler.write(f'// Raw data converting from DeNSSP to Cadence ({version})\n')
        tb_handler.write(f'// Testbench file for playing data sets in simulations\n')
        tb_handler.write(f'// -----------------------------------------------------------------\n')
        tb_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        tb_handler.write(f'// Original sampling rate fs = {fs / 1e3: .2f} kHz\n')
        tb_handler.write(f'// Size of input array: {size_input}\n')
        tb_handler.write(f'// -----------------------------------------------------------------\n\n')
        tb_handler.write(f'module data_player(\n')
        tb_handler.write(f'\tinput electrical EN,\n')
        tb_handler.write(f'\toutput electrical Vout,\n')
        if trigger_avaible:
            tb_handler.write(f'\toutput electrical Vtrg,\n')
        tb_handler.write(f'\toutput electrical CLK\n')
        tb_handler.write(f');\n\n')
        tb_handler.write(f'\tparameter real AVDD = 0.6;\n')
        tb_handler.write(f'\tparameter real AVSS = -0.6;\n')
        tb_handler.write(f'\tparameter real T_SAMP = {1/fs};\n\n')
        tb_handler.write(f'\treal v_cm, v_read, v_clk;\n')
        if trigger_avaible:
            tb_handler.write(f'\treal v_trgg;\n')
        tb_handler.write(f'\tinteger cnt_pos;\n\n')
        tb_handler.write(f'\tanalog begin\n')
        tb_handler.write(f'\t\t@(initial_step) begin\n')
        tb_handler.write(f'\t\t\tcnt_pos = 0;\n')
        tb_handler.write(f'\t\t\tv_clk = -1;\n')
        tb_handler.write(f'\t\t\t// Change path to txt-file\n')
        tb_handler.write(f'\t\t\tv_read = $fopen("/home/{username}/data_raw.txt", "r");\n')
        if trigger_avaible:
            tb_handler.write(f'\t\t\tv_trgg = $fopen("/home/{username}/data_trg.txt", "r");\n')
        tb_handler.write(f'\t\tend\n')
        tb_handler.write(f'\t\tv_cm = (AVDD - AVSS)/2;\n\n')
        tb_handler.write(f'\t\t// Clock generation and counter\n')
        tb_handler.write(f'\t\t@(timer(0, T_SAMP/2)) begin\n')
        tb_handler.write(f'\t\t\tif(V(EN) > v_cm) begin\n')
        tb_handler.write(f'\t\t\t\tv_clk = -v_clk;\n')
        tb_handler.write(f'\t\t\t\tcnt_pos = cnt_pos + 1;\n')
        tb_handler.write(f'\t\t\tend else begin\n')
        tb_handler.write(f'\t\t\t\tv_clk = -1;\n')
        tb_handler.write(f'\t\t\t\tcnt_pos = 1;\n')
        tb_handler.write(f'\t\t\tend\n')
        tb_handler.write(f'\t\tend\n\n')
        tb_handler.write(f'\t\t// Generate output\n')
        tb_handler.write(f'\t\tV(Vout) <+ v_read;\n')
        if trigger_avaible:
            tb_handler.write(f'\t\tV(Vtrg) <+ AVSS + (1+v_trg)/2* AVDD;\n')
        tb_handler.write(f'\t\tV(CLK) <+ AVSS + (1+v_clk)/2* AVDD;\n')
        tb_handler.write(f'\tend\n')
        tb_handler.write(f'endmodule\n')

    # --- Second file: Memory of rawdata ($readmemh)
    print('... transfer rawdata to txt file (*.txt)')
    with open(path2mem, 'w') as mem_handler:
        for idx, val in enumerate(raw_data0):
            mem_handler.write(f'{val}\n')

    # --- Third file: Memory of trigger ($readmemb)
    if trigger_avaible:
        print('... transfer trigger signals to txt file (*.txt)')
        with open(path2trg, 'w') as mem_handler:
            for idx, val in enumerate(trgg0):
                mem_handler.write(f'{val}\n')
