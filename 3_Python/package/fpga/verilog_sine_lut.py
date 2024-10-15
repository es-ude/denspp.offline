from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
from package.fpga.signal_type import generation_sinusoidal_waveform


def create_testbench(bitsize_lut: int,
                     f_sys: float, f_rpt: float, f_sine: float,
                     path2save='', num_periods=2) -> None:
    """Creating the testbench environment in Verilog for using in digital design software (frames)
    Args:
        bitsize_lut:    Used quantization level for generating sinusoidal waveform LUT
        f_sys:          System clock used in FPGA design
        f_rpt:          Frequency of the timer interrupt
        f_sine:         Target frequency of the sinusoidal waveform at output
        path2save:      Path for saving the verilog output files
        num_periods:    Number of periods
    Return:
        None
    """
    cnt_wait_val = int(f_sys / f_rpt)
    period_smp = int(1e9 / f_sine)
    period_clk = int(0.5 * 1e9 / f_sys)

    size_cnt_runs = int(period_smp/period_clk)
    if path2save != '' and not isdir(path2save):
        mkdir(path2save)

    file_content = [
        f'`timescale 1ns / 1ps',
        f'// --------------------------------------------------------------------------------------',
        f'// Company: UDE-ES',
        f'// Design Name: Testbench for running SineWFG emulator',
        f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}',
        f'// Target Devices: Simulation file',
        f'// Comments: Multply CNT_VAL_WAIT with (1 + 1/(size(sine_lutram)-1)) for right timing',
        f'// ------------------------------------------------------------------------------------\n',
        f'module TB_SineLUT();',
        f'\tlocalparam CLK_CYC_NS = \'d{period_clk}, CNT_VAL_WAIT = \'d{cnt_wait_val};',
        f'\tlocalparam CNT_PERIODS = \'d{size_cnt_runs}, NUM_PERIODS = \'d{num_periods};\n',
        f'\treg CLK_SYS, nRST, EN_LUT;',
        f'\treg [$clog2(CNT_VAL_WAIT):0] cnt_lut;',
        f'\twire [{bitsize_lut - 1}:0] SINE_LUT;\n',
        f'\tSineWFG_LUT#($clog2(CNT_VAL_WAIT)) DUT0(',
        f'\t\t.CLK_SYS(CLK_SYS),',
        f'\t\t.nRST(nRST),',
        f'\t\t.EN(EN_LUT),',
        f'\t\t.CNT_VAL(cnt_lut),',
        f'\t\t.SINE(SINE_LUT)',
        f'\t);\n',
        f'\t// Control scheme for getting the data dependent on the sampling clock',
        f'\talways begin',
        f'\t\t#(CLK_CYC_NS) CLK_SYS = ~CLK_SYS;',
        f'\tend\n',
        f'\tinitial begin',
        f'\t\tCLK_SYS = 1\'d0;',
        f'\t\tnRST = 1\'d1;',
        f'\t\tEN_LUT = 1\'d0;',
        f'\t\tcnt_lut = CNT_VAL_WAIT;\n',
        f'\t\t//Step #1: Reset-Phase',
        f'\t\t# (7* CLK_CYC_NS);   nRST <= 1\'b1;',
        f'\t\trepeat(2) begin',
        f'\t\t\t# (10* CLK_CYC_NS);   nRST <= 1\'b0;',
        f'\t\t\t# (10* CLK_CYC_NS);   nRST <= 1\'b1;',
        f'\t\tend\n',
        f'\t\t//Step #2: Enable SineLUT',
        f'\t\t#(10* CLK_CYC_NS);   EN_LUT = 1\'b1;\n',
        f'\t\t//Step #3: End simulation',
        f'\t\t#(NUM_PERIODS* CNT_PERIODS* CLK_CYC_NS) $stop;',
        f'\tend\n',
        f'endmodule'
    ]

    # --- Write to file
    print('... create testbench verilog file (*.v)')
    with open(join(path2save, f'TB_SineLUT.v'), 'w') as tb_handler:
        for line in file_content:
            tb_handler.write(line + '\n')
    tb_handler.close()


def generate_sinelut(bitsize_lut: int,
                     f_sys: float, f_rpt: float, f_sine: float, out_signed=False, do_optimized=False,
                     path2save='') -> None:
    """Generating Verilog file with SINE_LUT for sinusoidal waveform generation
    Args:
        bitsize_lut:    Used quantization level for generating sinusoidal waveform LUT
        f_sys:          System clock used in FPGA design
        f_rpt:          Frequency of the timer interrupt
        f_sine:         Target frequency of the sinusoidal waveform at output
        out_signed:     Decision if LUT values are signed [otherwise unsigned]
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    sine_lut = generation_sinusoidal_waveform(bitsize_lut, f_rpt, f_sine, do_optimized=do_optimized)

    # Bitwidth declaration
    num_cntsize = f_sys / f_rpt
    new_cnt_wait = num_cntsize * (1 + 1 / (sine_lut.size - 1))
    size_cnt_wait = int(np.ceil(np.log2(new_cnt_wait)))

    num_lutsine = (1.0 if not do_optimized else 0.25) * f_rpt / f_sine
    size_cnt_sine = int(np.ceil(np.log2(num_lutsine)))
    size_lut_sine = int(np.log2(sine_lut.size))
    size_ram = bitsize_lut - (1 if not do_optimized else 2)

    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    file_content = [
        f'`timescale 1ns / 1ps',
        f'// ---------------------------------------------------',
        f'// Company: UDE-ES',
        f'// Design Name: SineLUT Generator',
        f'// Generate file on: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}',
        f'// Original sinusoidal frequency = {f_sine / 1e3: .1f} kHz',
        f'// Size of SineLUT array: {sine_lut.size} ({size_lut_sine} bit)',
        f'// ---------------------------------------------------\n',
        f'module SineWFG_LUT#(',
        f'\tparameter CNT_VAL_SIZE = 6\'d{size_cnt_wait}',
        f')(',
        f'\tinput wire CLK_SYS,',
        f'\tinput wire nRST,',
        f'\tinput wire EN,',
        f'\tinput wire [CNT_VAL_SIZE-\'d1:0] CNT_VAL,'
        f'\toutput wire' + 'signed' if out_signed else '' + f'[{bitsize_lut - 1}:0] SINE',
        f');\n',
        f'localparam SIZE_SINE_LUT = 8\'d{sine_lut.size};\n',
        f'reg [1:0] cnt_phase;\n' if do_optimized else '',
        f'reg [{size_cnt_sine - 1}:0] cnt_sine;',
        f'reg [CNT_VAL_SIZE-\'d1:0] cnt_wait;',
        f'wire' + 'signed' if out_signed else '' + f'[{size_ram}:0] sine_lutram [{sine_lut.size - 1}:0];'
    ]

    # --- Generating verilog files
    print('... create verilog file for handling data (*.v)')
    with open(join(path2save, 'SINE_WFG.v'), 'w') as v_handler:
        for line in file_content:
            v_handler.write(line + '\n')

        if do_optimized and not out_signed:
            v_handler.write('assign SINE = (cnt_phase == 2\'d0) ? {1\'d1, sine_lutram[cnt_sine]} : \n')
            v_handler.write('\t\t\t (cnt_phase == 2\'d1) ? {1\'d1, sine_lutram[SIZE_SINE_LUT-cnt_sine-\'d1]} : \n')
            v_handler.write('\t\t\t (cnt_phase == 2\'d2) ? {1\'d0, {(' + str(bitsize_lut - 1) + '){1\'d1}}} - sine_lutram[cnt_sine] : \n'
                            '\t\t\t {1\'d0, {(' + str(bitsize_lut - 1) + '){1\'d1}}} - sine_lutram[SIZE_SINE_LUT-cnt_sine-\'d1];\n\n')
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
    v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'
    n_bit = 8
    f_sys = 100e6
    f_rpt = 200e3
    f_sine = 1e3

    print("\nCreating the verilog files\n====================================")
    create_testbench(n_bit, f_sys, f_rpt, f_sine, path2save=path2save)
    generate_sinelut(n_bit, f_sys, f_rpt, f_sine, do_optimized=True, path2save=path2save)
