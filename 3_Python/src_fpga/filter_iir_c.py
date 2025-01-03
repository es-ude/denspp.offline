from os import mkdir
from shutil import copyfile
from os.path import join, isdir
from datetime import datetime

from fpga.helper.emulator_filter import filter_stage
from fpga.helper.translater import (get_embedded_datatype, replace_variables_with_parameters,
                                    generate_params_list)


def generate_iir_filter_files(data_bitsize: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, filter_corner: list,
                              filter_btype='low', filter_ftype='butter', filter_id='',
                              file_name='filter_iir', path2save='', define_path='src') -> None:
    """Generating C files for IIR filtering on microcontroller
    Args:
        data_bitsize:   Used quantization level for data stream
        data_signed:    Decision if LUT values are signed [otherwise unsigned]
        filter_id:      ID of used filter
        filter_order:   Order of the filter
        sampling_rate:  Sampling clock of data stream processing
        filter_corner:  list with corner frequency for used filter
        filter_btype:   Used filter type ['low', 'high', 'bandpass', 'bandstop', 'all' (only 1st, 2nd order)]
        filter_ftype:   Used filter design ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    module_id_used = filter_id if filter_id else '0'
    filter_emulator = filter_stage(filter_order, sampling_rate, filter_corner, True,
                                   ftype=filter_ftype, btype=filter_btype)
    filter_coeff = filter_emulator.get_coeff_full()
    filter_coeff_used = ', '.join(map(str, filter_coeff['coeffa'])) + ', ' + ', '.join(map(str, filter_coeff['coeffb']))

    data_type_filter = get_embedded_datatype(data_bitsize, data_signed)
    template_c = __generate_filter_iir_template()
    params = {
        'datetime_created':     datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'path2include':         define_path,
        'template_name':        'filter_iir_template.h',
        'device_id':            module_id_used.upper(),
        'data_type':            data_type_filter,
        'fs':                   f'{sampling_rate}',
        'filter_type':          f'{filter_btype}, {filter_ftype}',
        'filter_corner':        ', '.join(map(str, filter_corner)),
        'filter_order':         str(filter_order),
        'coeff_order':          str(filter_coeff['coeffa'].size),
        'tap_order':            str(filter_coeff['coeffa'].size-1),
        'coeffs_string':        filter_coeff_used

    }
    proto_file = replace_variables_with_parameters(template_c['head'], params)
    imple_file = replace_variables_with_parameters(template_c['func'], params)

    # --- Write lines to file
    used_template_name = params["template_name"]
    copyfile(f'template_c/{used_template_name}', join(path2save, f'{used_template_name}'))
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    # --- Write lines to file
    with open(join(path2save, f'{file_name}{module_id_used.lower()}.h'), 'w') as v_handler:
        for line in proto_file:
            v_handler.write(line + '\n')
    v_handler.close()

    with open(join(path2save, f'{file_name}{module_id_used.lower()}.c'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line + '\n')
    v_handler.close()


def __generate_filter_iir_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate an IIR filter on MCUs
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        '// --- Generating an IIR filter template (Direct Form II)',
        '// Copyright @ UDE-IES',
        '// Code generated on: {$datetime_created}',
        '// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})',
        '// Used filter coefficient order (b_0, b_1, b_2, ..., b_N)',
        '# include "{$path2include}/{$template_name}"',
        'DEF_NEW_IIR_FILTER_PROTO({$device_id}, {$data_type})'
    ]
    func_temp = [
        '// --- Generating an IIR filter template (Direct Form II)',
        '// Copyright @ UDE-IES',
        '// Code generated on: {$datetime_created}',
        '// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})',
        '// Used filter coefficient order (a_0, a_1, ... a_N, b_0, b_1, ..., b_N)',
        '# include "{$path2include}/{$template_name}"',
        'DEF_NEW_IIR_FILTER_IMPL({$device_id}, {$data_type}, {$coeff_order}, {$tap_order}, {$coeffs_string})'
    ]

    # --- Generate list with all metrics
    params_temp = generate_params_list(header_temp)
    params_temp = generate_params_list(func_temp, params_temp)

    return {'head': header_temp, 'func': func_temp, 'params': params_temp}


if __name__ == '__main__':
    path2save_out = '../../runs'

    # generate_iir_filter_files(16, True, 0, 2, 1e3, [100], path2save=path2save_out)
    generate_iir_filter_files(16, True, 2, 1e3, [100],
                              filter_btype='low', path2save=path2save_out, filter_id='0')
    generate_iir_filter_files(16, True, 2, 1e3, [100],
                              filter_btype='high', path2save=path2save_out, filter_id='1')
    generate_iir_filter_files(16, True, 1, 1e3, [100],
                              filter_btype='all', path2save=path2save_out, filter_id='2')
    generate_iir_filter_files(16, True, 1, 1e3, [10, 100],
                              filter_btype='bandpass', path2save=path2save_out, filter_id='3')
    generate_iir_filter_files(16, True, 2, 1e3, [10, 100],
                              filter_btype='bandpass', path2save=path2save_out, filter_id='4')
    generate_iir_filter_files(16, True, 2, 1e3, [10, 100],
                              filter_btype='bandstop', path2save=path2save_out, filter_id='5')
