from os import mkdir
from shutil import copyfile
from os.path import join, isdir
from datetime import datetime

from package.fpga.helper.translater import (get_embedded_datatype, replace_variables_with_parameters,
                                            generate_params_list)


def generate_moving_avg_files(data_bitsize: int, data_signed: bool,
                              filter_order: int, sampling_rate: float,
                              module_id='', file_name='filter_mavg',
                              path2save='', define_path='src') -> None:
    """Generating C files for moving average on microcontroller
    Args:
        data_bitsize:   Used quantization level for data stream
        data_signed:    Decision if LUT values are signed [otherwise unsigned]
        module_id:      ID of used filter structure
        filter_order:   Order of the filter
        sampling_rate:  Sampling clock of data stream processing
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    module_id_used = module_id if module_id else '0'
    data_type_filter = get_embedded_datatype(data_bitsize, data_signed)
    template_c = __generate_filter_mavg_template()

    params = {
        'datetime_created':     datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'path2include':         define_path,
        'template_name':        'filter_mavg_template.h',
        'device_id':            module_id_used.upper(),
        'data_type':            data_type_filter,
        'fs':                   f'{sampling_rate}',
        'filter_order':         str(filter_order),
        'filter_coeff':         str(1/filter_order)
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


def __generate_filter_mavg_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate a moving average FIR on MCUs
    Args:
        None
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        f'// --- Generating a moving average',
        '// Copyright @ UDE-IES',
        '// Code generated on: {$datetime_created}',
        '// Params: N = {$filter_order}, f_s = {$fs} Hz',
        '# include "{$path2include}/{$template_name}"',
        'DEF_NEW_MAVG_FILTER_PROTO({$device_id}, {$data_type})'
    ]
    func_temp = [
        f'// --- Generating a moving average filter',
        '// Copyright @ UDE-IES',
        '// Code generated on: {$datetime_created}',
        '// Params: N = {$filter_order}, f_s = {$fs} Hz',
        '# include "{$path2include}/{$template_name}"',
        'DEF_NEW_MAVG_FILTER_IMPL({$device_id}, {$data_type}, {$filter_order}, {$filter_coeff})'
    ]

    # --- Generate list with all metrics
    params_temp = generate_params_list(header_temp)
    params_temp = generate_params_list(func_temp, params_temp)

    return {'head': header_temp, 'func': func_temp, 'params': params_temp}


if __name__ == '__main__':
    path2save = '../../runs'

    generate_moving_avg_files(14, True, 21, 1e3,  path2save=path2save)
