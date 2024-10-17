from os import mkdir
from os.path import join, isdir, abspath
from datetime import datetime
import numpy as np

from package.fpga.signal_type import generation_sinusoidal_waveform
from package.fpga.c_helper import (get_embedded_datatype,
                                   slicing_data_intro_string_array, replace_variables_with_parameters)


def generate_lut_files(bitsize_lut: int, f_sys: float, f_rpt: float, f_sine: float,
                       out_signed=False, do_optimized=False, slice_val=4,
                       file_name='waveform_lut', path2save='') -> None:
    """Generating C file with SINE_LUT for sinusoidal waveform generation
    Args:
        bitsize_lut:    Used quantization level for generating sinusoidal waveform LUT
        f_sys:          System clock used in FPGA design
        f_rpt:          Frequency of the timer interrupt
        f_sine:         Target frequency of the sinusoidal waveform at output
        out_signed:     Decision if LUT values are signed [otherwise unsigned]
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        slice_val:      Slicing the LUT into defined numbers in each row [Default: 4]
        file_name:      File name of generated output
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    # --- Step #1: Building template and data
    datatype_data_ext = get_embedded_datatype(bitsize_lut, out_signed)
    bitwidth_mcu = int(datatype_data_ext.split('int')[-1].split('_')[0])

    sine_lut_sliced = slicing_data_intro_string_array(
        parameter=generation_sinusoidal_waveform(bitsize_lut, f_rpt, f_sine, bitwidth_mcu, do_optimized=do_optimized),
        slice_value=slice_val
    )
    template = generate_waveform_lut_template_c(do_optimized)

    # --- Step #2: Generating the values for parameter dict
    num_lutsine = int((1.0 if not do_optimized else 0.25) * f_rpt / f_sine)
    size_lut_sine = int(np.log2(num_lutsine))
    parameters = {
        'datatype_cnt':     get_embedded_datatype(size_lut_sine, out_signed=False),
        'datatype_int':     get_embedded_datatype(bitsize_lut),
        'datatype_ext':     datatype_data_ext,
        'num_cntsize':      str(int(f_sys / f_rpt)),
        'num_lutsine':      str(num_lutsine),
        'lut_offset':       str(0 if out_signed else (2 ** (bitwidth_mcu - 1))),
        'lut_data':         sine_lut_sliced,
        'lut_type':         'sinusoidal'
    }

    # --- Step #3: Replace string parameters with real values
    head = replace_variables_with_parameters(template['head'], parameters)
    func = replace_variables_with_parameters(template['func'], parameters)

    # --- Write lines to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    print("\nCreating the C files\n====================================")
    with open(join(path2save, f'{file_name}.h'), 'w') as v_handler:
        for line in head:
            v_handler.write(line + '\n')
    v_handler.close()

    with open(join(path2save, f'{file_name}.c'), 'w') as v_handler:
        for line in func:
            v_handler.write(line + '\n')
    v_handler.close()
    print(f'... created C (*.h + *.c) in: {abspath(path2save)}')


def generate_waveform_lut_template_c(do_full_opt: bool) -> dict:
    """Generating the C template for generating the waveforms
    Args:
        do_full_opt:    Get C functions for full LUT representation or reduced LUT with mirror techniques
    Return:
        Dictionary with files
    """
    params = ['datatype_cnt', 'datatype_int', 'datatype_ext',
              'num_cntsize', 'num_lutsine', 'lut_offset', 'lut_data', 'lut_type']

    # --- Generating the content of the header file
    header_content = [
        '{$datatype_cnt} read_waveform_position(void);',
        '{$datatype_cnt} read_waveform_length(void);',
        'uint8_t read_waveform_state(void);',
        '{$datatype_ext} read_waveform_value_runtime(void);'
    ]

    # --- Generating the content of the C file
    func_content = [
        '#include <stdint.h>\n',
        '#define ISR_WAVEFORM_TMR_VAL {$num_cntsize};',
        '{$datatype_cnt} waveform_lut_state = 0;',
        '{$datatype_cnt} waveform_lut_cnt = 0;\n',
        '// --- Implementing {$lut_type} waveform as LUT',
        '{$datatype_cnt} waveform_lut_length = {$num_lutsine};',
        '{$datatype_ext} waveform_lut_data[{$num_lutsine}] = {',
        '{$lut_data}'
        '};\n'
    ]
    call_lut_normal = [
        '{$datatype_cnt} read_waveform_position(void){',
        '\treturn waveform_lut_cnt;',
        '};',
        '{$datatype_cnt} read_waveform_length(void){',
        '\treturn waveform_lut_length;',
        '};',
        'uint8_t read_waveform_state(void){',
        '\treturn waveform_lut_state;',
        '};\n',
        '{$datatype_ext} read_waveform_value_runtime(void){',
        '\t{$datatype_ext} data = waveform_lut_data[waveform_lut_cnt];',
        '\tif(waveform_lut_cnt == waveform_lut_length -1){',
        '\t\twaveform_lut_cnt = 0;',
        '\t\twaveform_lut_state = 0;',
        '\t} else {',
        '\t\twaveform_lut_cnt++;',
        '\t\twaveform_lut_state = 1;',
        '\t};',
        '\treturn data;',
        '};'
    ]
    call_lut_mirror = [
        f'{header_content[0][:-1]}{{',
        '\treturn waveform_lut_cnt;',
        '};',
        f'{header_content[1][:-1]}{{',
        '\treturn waveform_lut_length;',
        '};',
        f'{header_content[2][:-1]}{{',
        '\treturn waveform_lut_state;',
        '};\n',
        f'{header_content[3][:-1]}{{',
        '\t{$datatype_int} data = 0;',
        '\tif(waveform_lut_state == 0){',
        '\t\tdata = waveform_lut_data[waveform_lut_cnt];',
        '\t\twaveform_lut_cnt++;',
        '\t\tif(waveform_lut_cnt == (waveform_lut_length - 1)){',
        '\t\t\twaveform_lut_state = 1;',
        '\t\t} else {',
        '\t\t\twaveform_lut_state = 0;',
        '\t\t};',
        '\t} else if(waveform_lut_state == 1){',
        '\t\tdata = waveform_lut_data[waveform_lut_cnt];',
        '\t\twaveform_lut_cnt--;',
        '\t\tif(waveform_lut_cnt == 0){{',
        '\t\t\twaveform_lut_state = 2;',
        '\t\t} else {',
        '\t\t\twaveform_lut_state = 1;',
        '\t\t};',
        '\t} else if(waveform_lut_state == 2){',
        '\t\tdata = -waveform_lut_data[waveform_lut_cnt];',
        '\t\twaveform_lut_cnt++;',
        '\t\tif(waveform_lut_cnt == (waveform_lut_length - 1)){',
        '\t\t\twaveform_lut_state = 3;',
        '\t\t} else {',
        '\t\t\twaveform_lut_state = 2;',
        '\t\t};',
        '\t} else if(waveform_lut_state == 3){',
        '\t\tdata = -waveform_lut_data[waveform_lut_cnt];',
        '\t\twaveform_lut_cnt--;',
        '\t\tif(waveform_lut_cnt == 0){',
        '\t\t\twaveform_lut_state = 0;',
        '\t\t} else {',
        '\t\t\twaveform_lut_state = 3;',
        '\t\t};',
        '\t} else{',
        '\t\twaveform_lut_state = 0;',
        '\t\twaveform_lut_cnt = 0;',
        '\t};',
        '\treturn {$lut_offset} + data;',
        '};'
    ]
    [func_content.append(line) for line in (call_lut_mirror if do_full_opt else call_lut_normal)]
    return {'head': header_content, 'func': func_content, 'params': params, 'impl': 'C'}


if __name__ == '__main__':
    path2save = '../../runs'
    n_bit = 14
    f_sys = 32e6
    f_rpt = 250e3
    f_sine = 10e3

    generate_lut_files(n_bit, f_sys, f_rpt, f_sine, do_optimized=False, out_signed=True, path2save=path2save)
