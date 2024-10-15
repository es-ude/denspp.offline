from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
from package.fpga.signal_type import generation_sinusoidal_waveform


def get_embedded_datatype(bits_used: int, out_signed=True) -> str:
    """Determine the C datatype for processing data
    Args:
        bits_used:      Integer with bit value
        out_signed:     Boolean if datatype is signed or unsigned
    Return:
        String with datatype in C
    """
    # Define datatyp used in embedded device
    bit_compare = np.array((8, 16, 32, 64))
    used_bitval = np.argwhere((bit_compare / bits_used) - 1 >= 0).flatten()[0]
    return ('' if out_signed else 'u') + 'int' + f'{bit_compare[used_bitval]}' + f'_t'


def generate_sinelut(bitsize_lut: int,
                     f_sys: float, f_rpt: float, f_sine: float, out_signed=False, do_optimized=False,
                     path2save='') -> None:
    """Generating C file with SINE_LUT for sinusoidal waveform generation
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
    num_cntsize = int(f_sys / f_rpt)
    reduced_samples = 1.0 if not do_optimized else 0.25
    num_lutsine = reduced_samples * f_rpt / f_sine
    size_lut_sine = int(np.log2(sine_lut.size))

    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    # Define datatyp used in embedded device
    used_datatype_cnt = get_embedded_datatype(size_lut_sine, out_signed=False)
    used_datatype_data_ext = get_embedded_datatype(bitsize_lut, out_signed)
    used_datatype_data_int = get_embedded_datatype(bitsize_lut)
    bitwidth_mcu = int(used_datatype_data_ext.split('int')[-1].split('_')[0])
    sine_lut = generation_sinusoidal_waveform(bitsize_lut, f_rpt, f_sine, bitwidth_mcu, do_optimized=do_optimized)

    # Generating content for C file
    file_content = [
        f'#include <stdint.h>\n',
        f'#define ISR_WAVEFORM_TMR_VAL {num_cntsize};',
        f'{used_datatype_cnt} waveform_lut_state = 0;',
        f'{used_datatype_cnt} waveform_lut_cnt = 0;',
        f'{used_datatype_data_ext} waveform_lut_length = {int(num_lutsine)};',
        f'{used_datatype_data_ext} waveform_lut_data[{int(num_lutsine)}] = {{'
    ]

    # Bringing data into right format
    slice_val = 4
    num_ite = int(np.ceil(sine_lut.size / slice_val))
    for idx0 in range(num_ite):
        used_string = ''
        data_row = sine_lut[idx0*slice_val:(idx0+1)*slice_val] if (idx0+1)*slice_val < sine_lut.size else sine_lut[idx0*slice_val:]
        for idx1, val0 in enumerate(data_row):
            used_string += f'\t{val0}' + (',' if not (idx0*4) + idx1 == sine_lut.size-1 else '')
        file_content.append(used_string)
    file_content.append('};\n')

    # Function call
    function_call0 = [
        f'{used_datatype_cnt} read_waveform_position(void){{',
        f'\treturn waveform_lut_cnt;',
        f'}};\n',
        f'uint8_t read_waveform_state(void){{',
        f'\treturn waveform_lut_state;',
        f'}};\n',
        f'{used_datatype_data_ext} read_waveform_value_runtime(void){{',
        f'\t{used_datatype_data_int} data = 0;',
        f'\tif(waveform_lut_state == 0){{',
        f'\t\tdata = waveform_lut_data[waveform_lut_cnt];',
        f'\t\twaveform_lut_cnt++;',
        f'\t\tif(waveform_lut_cnt == (waveform_lut_length - 1)){{',
        f'\t\t\twaveform_lut_state = 1;',
        f'\t\t}} else {{',
        f'\t\t\twaveform_lut_state = 0;',
        f'\t\t}};',
        f'\t}} else if(waveform_lut_state == 1){{',
        f'\t\tdata = waveform_lut_data[waveform_lut_cnt];',
        f'\t\twaveform_lut_cnt--;',
        f'\t\tif(waveform_lut_cnt == 0){{',
        f'\t\t\twaveform_lut_state = 2;',
        f'\t\t}} else {{',
        f'\t\t\twaveform_lut_state = 1;',
        f'\t\t}};',
        f'\t}} else if(waveform_lut_state == 2){{',
        f'\t\tdata = -waveform_lut_data[waveform_lut_cnt];',
        f'\t\twaveform_lut_cnt++;',
        f'\t\tif(waveform_lut_cnt == (waveform_lut_length - 1)){{',
        f'\t\t\twaveform_lut_state = 3;',
        f'\t\t}} else {{',
        f'\t\t\twaveform_lut_state = 2;',
        f'\t\t}};',
        f'\t}} else if(waveform_lut_state == 3){{',
        f'\t\tdata = -waveform_lut_data[waveform_lut_cnt];',
        f'\t\twaveform_lut_cnt--;',
        f'\t\tif(waveform_lut_cnt == 0){{',
        f'\t\t\twaveform_lut_state = 0;',
        f'\t\t}} else {{',
        f'\t\t\twaveform_lut_state = 3;',
        f'\t\t}};',
        f'\t}} else{{',
        f'\t\twaveform_lut_state = 0;',
        f'\t\twaveform_lut_cnt = 0;',
        f'\t}};',
        f'\treturn data;' if out_signed else f'\treturn {(2 ** (bitsize_lut-1))} + data;',
        f'}};'
    ]

    function_call1 = [
        f'{used_datatype_cnt} read_waveform_position(void){{',
        f'\treturn waveform_lut_cnt;',
        f'}};\n',
        f'uint8_t read_waveform_state(void){{',
        f'\treturn waveform_lut_state;',
        f'}};\n',
        f'{used_datatype_data_ext} read_waveform_value_runtime(void){{',
        f'\t{used_datatype_data_ext} data = waveform_lut_data[waveform_lut_cnt];',
        f'\tif(waveform_lut_cnt == waveform_lut_length -1){{',
        f'\t\twaveform_lut_cnt = 0;',
        f'\t\twaveform_lut_state = 0;',
        f'\t}} else {{',
        f'\t\twaveform_lut_cnt++;',
        f'\t\twaveform_lut_state = 1;',
        f'\t}};',
        f'\treturn data;',
        f'}};'
    ]

    header_call = [
        f'{used_datatype_cnt} read_waveform_position(void);',
        f'uint8_t read_waveform_state(void);',
        f'{used_datatype_data_ext} read_waveform_value_runtime(void);'
    ]

    # --- Write lines to file
    print('... create C file for handling data (*.c)')
    with open(join(path2save, 'SINE_WFG.c'), 'w') as v_handler:
        for line in file_content:
            v_handler.write(line + '\n')
        list2write = function_call0 if do_optimized else function_call1
        for line in list2write:
            v_handler.write(line + '\n')
    v_handler.close()

    print('... create C file for handling data (*.h)')
    with open(join(path2save, 'SINE_WFG.h'), 'w') as v_handler:
        for line in header_call:
            v_handler.write(line + '\n')
    v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'
    n_bit = 14
    f_sys = 32e6
    f_rpt = 250e3
    f_sine = 10e3

    print("\nCreating the verilog files\n====================================")
    generate_sinelut(n_bit, f_sys, f_rpt, f_sine, out_signed=True, do_optimized=False, path2save=path2save)
