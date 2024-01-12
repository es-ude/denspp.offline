from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
from fxpmath import Fxp

def create_testbench(bitsize_sine: int,
                     f_sys: float, f_rpt: float, f_sine: float,
                     path2save='') -> None:
    """Creating the testbench environment in Verilog for using in digital design software (frames)"""
    num_periods = 2
    cnt_wait_val = int(f_sys / f_rpt)
    period_smp = int(1e9 / f_sine)
    period_clk = int(0.5 * 1e9 / f_sys)

    size_cnt_runs = int(period_smp/period_clk)
    if path2save != '' and not isdir(path2save):
        mkdir(path2save)

    print('... create testbench verilog file (*.v)')
    with open(join(path2save, f'TB_SineLUT.v'), 'w') as tb_handler:
        tb_handler.write(f'`timescale 1ns / 1ps\n')
        tb_handler.write(f'// --------------------------------------------------------------------------------------\n')
        tb_handler.write(f'// Company: UDE-ES\n')
        tb_handler.write(f'// Design Name: Testbench for running SineWFG emulator\n')
        tb_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        tb_handler.write(f'// Target Devices: Simulation file\n')
        tb_handler.write(f'// Comments: Multply CNT_VAL_WAIT with (1 + 1/(size(sine_lutram)-1)) for right timing\n')
        tb_handler.write(f'// ------------------------------------------------------------------------------------\n\n')

        tb_handler.write(f'module TB_SineLUT();\n')
        tb_handler.write(f'\tlocalparam CLK_CYC_NS = \'d{period_clk}, CNT_VAL_WAIT = \'d{cnt_wait_val};\n'
                         f'\tlocalparam CNT_PERIODS = \'d{size_cnt_runs}, NUM_PERIODS = \'d{num_periods};\n\n')
        tb_handler.write(f'\treg CLK_SYS, nRST, EN_LUT;\n')
        tb_handler.write(f'\treg [$clog2(CNT_VAL_WAIT):0] cnt_lut;\n')
        tb_handler.write(f'\twire [{bitsize_sine - 1}:0] SINE_LUT;\n\n')

        tb_handler.write(f'\tSineWFG_LUT#($clog2(CNT_VAL_WAIT)) DUT0(\n')
        tb_handler.write(f'\t\t.CLK_SYS(CLK_SYS),\n')
        tb_handler.write(f'\t\t.nRST(nRST),\n')
        tb_handler.write(f'\t\t.EN(EN_LUT),\n')
        tb_handler.write(f'\t\t.CNT_VAL(cnt_lut),\n')
        tb_handler.write(f'\t\t.SINE(SINE_LUT)\n')
        tb_handler.write(f'\t);\n\n')

        tb_handler.write(f'\t// Control scheme for getting the data dependent on the sampling clock\n')
        tb_handler.write(f'\talways begin\n')
        tb_handler.write(f'\t\t#(CLK_CYC_NS) CLK_SYS = ~CLK_SYS;\n')
        tb_handler.write(f'\tend\n\n')

        tb_handler.write(f'\tinitial begin\n')
        tb_handler.write(f'\t\tCLK_SYS = 1\'d0;\n')
        tb_handler.write(f'\t\tnRST = 1\'d1;\n')
        tb_handler.write(f'\t\tEN_LUT = 1\'d0;\n')
        tb_handler.write(f'\t\tcnt_lut = CNT_VAL_WAIT;\n\n')

        tb_handler.write(f'\t\t//Step #1: Reset-Phase\n')
        tb_handler.write(f'\t\t# (7* CLK_CYC_NS);   nRST <= 1\'b1;\n')
        tb_handler.write(f'\t\trepeat(2) begin\n')
        tb_handler.write(f'\t\t\t# (10* CLK_CYC_NS);   nRST <= 1\'b0;\n')
        tb_handler.write(f'\t\t\t# (10* CLK_CYC_NS);   nRST <= 1\'b1;\n')
        tb_handler.write(f'\t\tend\n\n')

        tb_handler.write(f'\t\t//Step #2: Enable SineLUT\n')
        tb_handler.write(f'\t\t#(10* CLK_CYC_NS);   EN_LUT = 1\'b1;\n\n')

        tb_handler.write(f'\t\t//Step #3: End simulation\n')
        tb_handler.write(f'\t\t#(NUM_PERIODS* CNT_PERIODS* CLK_CYC_NS) $stop;\n')
        tb_handler.write(f'\tend\n\n')
        tb_handler.write(f'endmodule')


def generate_sinelut(output_bitsize: int,
                     f_clk: float, f_rpt: float, f_sine: float,
                     path2save='data/fpga',
                     out_signed=False, do_optimized=False) -> None:
    """Generating Verilog file with SINE_LUT for sinusoidal waveform generator"""

    reduced_samples = 1.0 if not do_optimized else 0.25
    num_cntsize = f_clk / f_rpt
    num_lutsine = reduced_samples * f_rpt / f_sine
    size_ram = output_bitsize - (1 if not do_optimized else 2)

    # Generating sine waveform as template
    x0 = np.linspace(0, reduced_samples/f_sine, int(num_lutsine), endpoint=True)
    offset = 0 if do_optimized else 1
    sine_lut = (2 ** (output_bitsize-1) * (offset + np.sin(2 * np.pi * f_sine * x0)))
    if sine_lut.max() > (2 ** (output_bitsize-1)-1):
        xpos = np.argmax(sine_lut)
        sine_lut[xpos] = (2 ** (output_bitsize-1)-1)
    if sine_lut.min() < 0:
        xpos = np.argmin(sine_lut)
        sine_lut[xpos] = 0
    sine_lut = np.array(sine_lut, dtype=np.int32)

    # Bitwidth declaration
    new_cnt_wait = num_cntsize * (1 + 1 / (sine_lut.size - 1))
    size_cnt_wait = int(np.ceil(np.log2(new_cnt_wait)))
    size_cnt_sine = int(np.ceil(np.log2(num_lutsine)))
    size_lut_sine = int(np.log2(sine_lut.size))

    # Checking if path is available
    if path2save != '' and not isdir(path2save):
        mkdir(path2save)

    # Generating verilog files
    print('... create verilog file for handling data (*.v)')
    with open(join(path2save, 'SINE_WFG.v'), 'w') as v_handler:
        # --- Header 1: Used tools and data
        v_handler.write(f'`timescale 1ns / 1ps\n')
        v_handler.write(f'// ---------------------------------------------------\n')
        v_handler.write(f'// Company: UDE-ES\n')
        v_handler.write(f'// Design Name: SineLUT Generator\n')
        v_handler.write(f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
        v_handler.write(f'// Original sinusoidal frequency = {f_sine/1e3: .1f} kHz\n')
        v_handler.write(f'// Size of SineLUT array: {sine_lut.size} ({size_lut_sine} bit)\n')
        v_handler.write(f'// ---------------------------------------------------\n\n')

        # --- Code: Module initialization
        v_handler.write(f'module SineWFG_LUT#(\n'
                        f'\tparameter CNT_VAL_SIZE = 6\'d{size_cnt_wait}\n'
                        f')(\n')
        v_handler.write(f'\tinput wire CLK_SYS,\n')
        v_handler.write(f'\tinput wire nRST,\n')
        v_handler.write(f'\tinput wire EN,\n')
        v_handler.write(f'\tinput wire [CNT_VAL_SIZE-\'d1:0] CNT_VAL,\n')
        if out_signed:
            v_handler.write(f'\toutput wire signed [{output_bitsize-1}:0] SINE\n')
        else:
            v_handler.write(f'\toutput wire [{output_bitsize-1}:0] SINE\n')
        v_handler.write(f');\n\n')

        # CNT declaration
        v_handler.write(f'localparam SIZE_SINE_LUT = 8\'d{sine_lut.size};\n\n')
        if do_optimized:
            v_handler.write(f'reg [1:0] cnt_phase;\n')
        v_handler.write(f'reg [{size_cnt_sine-1}:0] cnt_sine;\n')
        v_handler.write(f'reg [CNT_VAL_SIZE-\'d1:0] cnt_wait;\n')

        # Output declaration
        if out_signed:
            v_handler.write(f'wire signed [{size_ram}:0] sine_lutram [{sine_lut.size-1}:0];\n')
        else:
            v_handler.write(f'wire [{size_ram}:0] sine_lutram [{sine_lut.size - 1}:0];\n')
        if do_optimized:
            v_handler.write('assign SINE = (cnt_phase == 2\'d0) ? {1\'d1, sine_lutram[cnt_sine]} : \n')
            v_handler.write('\t\t\t (cnt_phase == 2\'d1) ? {1\'d1, sine_lutram[SIZE_SINE_LUT-cnt_sine-\'d1]} : \n')
            v_handler.write('\t\t\t (cnt_phase == 2\'d2) ? {1\'d1, {(' + str(output_bitsize-1) + '){1\'d0}}} - sine_lutram[cnt_sine] : \n'
                            '\t\t\t {1\'d1, {(' + str(output_bitsize-1) + '){1\'d0}}} - sine_lutram[SIZE_SINE_LUT-cnt_sine-\'d1];\n\n')
        else:
            v_handler.write(f'assign SINE = sine_lutram[cnt_sine];\n\n')

        # Control Routine
        v_handler.write(f'//--- Counter\n')
        v_handler.write(f'always@(posedge CLK_SYS) begin\n')
        v_handler.write(f'\tif(~(nRST && EN)) begin\n')
        if do_optimized:
            v_handler.write(f'\t\tcnt_phase <= 2\'d0;\n')
        v_handler.write(f'\t\tcnt_sine <= {size_cnt_sine}\'d0;\n')
        v_handler.write(f'\t\tcnt_wait <= \'d0;\n')
        v_handler.write(f'\tend else begin\n')
        v_handler.write(f'\t\tif(cnt_wait == CNT_VAL) begin\n')
        if do_optimized:
            v_handler.write(
                f'\t\t\tcnt_phase <= cnt_phase + ((cnt_sine == {size_cnt_sine}\'d{sine_lut.size - 1}) ? 2\'d1 : 2\'d0);\n')
        v_handler.write(f'\t\t\tcnt_sine <= (cnt_sine == {size_cnt_sine}\'d{sine_lut.size-1}) ? {size_cnt_sine}\'d1 : cnt_sine + {size_cnt_sine}\'d1;\n')
        v_handler.write(f'\t\t\tcnt_wait <= \'d0;\n')
        v_handler.write(f'\t\tend else begin\n')
        if do_optimized:
            v_handler.write(f'\t\t\tcnt_phase <= cnt_phase;\n')
        v_handler.write(f'\t\t\tcnt_sine <= cnt_sine;\n')
        v_handler.write(f'\t\t\tcnt_wait <= cnt_wait + \'d1;\n')
        v_handler.write(f'\t\tend\n')
        v_handler.write(f'\tend\n')
        v_handler.write(f'end\n\n')

        # --- Code: Data values
        v_handler.write(f'// --- Data save in BRAM\n')
        for idx, val in enumerate(sine_lut):
            if val >= 0:
                out_string = f'{size_ram+1}\'d{val};\n'
            else:
                out_string = f'-{size_ram+1}\'d{np.abs(val)};\n'

            v_handler.write(f'assign sine_lutram[{idx}] = ' + out_string)
        # --- End of module
        v_handler.write(f'\nendmodule\n')


if __name__ == '__main__':
    path2save = 'C://Users//erbsloeh//Desktop'
    n_bit = 7
    f_sys = 100e6
    f_rpt = 200e3
    f_sine = 1e3
    create_testbench(n_bit, f_sys, f_rpt, f_sine, path2save)
    generate_sinelut(n_bit, f_sys, f_rpt, f_sine, path2save, do_optimized=True)
