from os import mkdir, getcwd
from os.path import join, isdir
from numpy import log2, ceil
from datetime import datetime
from fxpmath import Fxp
from shutil import copyfile

from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_moving_avg_files(data_bitsize: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, module_id='',
                              mode_multiplier=0, copy_testbench=False, file_name='filter_mavg', path2save='') -> None:
    """Generating Verilog files for moving average on FPGAs/ASICs
    Args:
        data_bitsize:       Used quantization level for data stream
        data_signed:        Decision if data stream in-/output values are signed [otherwise unsigned]
        filter_order:       Filter order of moving average filter
        sampling_rate:      Sampling clock of data stream processing
        module_id:          ID of used filter structure
        mode_multiplier:    Mode of multiplier (0= DSP slice from FPGA, 1= LUT Multiplier, 2= Ext. multiplier)
        copy_testbench:     Copy the template testbench file to output folder
        file_name:          Name of the generated files
        path2save:          Path for saving the verilog output files
    Return:
        None
    """
    module_id_used = module_id if module_id else '0'
    # --- Getting the design template
    path2template = join(getcwd(), 'template_verilog/filter_mavg_template.v')
    template_file = read_template_design_file(path2template)
    number_used = Fxp(1/filter_order, signed=False, n_word=data_bitsize, n_frac=data_bitsize)

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':    datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'use_own_mult':         '' if mode_multiplier == 1 else '//',
        'use_ext_mult':         '' if mode_multiplier == 2 else '//',
        'device_id':            module_id_used.upper(),
        'bitwidth_data':        str(data_bitsize),
        'length_mavg':          str(filter_order),
        'sampling_rate':        f'{sampling_rate}',
        'signed_data':          '0' if data_signed else '1'
    }

    # --- Write new design to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    if mode_multiplier == 1:
        copyfile('template_verilog/mult_lut_signed.v', f'{path2save}/mult_lut_signed.v')

    # Design Files
    imple_file = replace_variables_with_parameters(template_file['func'], params)
    with open(join(path2save, f'{file_name}{module_id_used.lower()}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()

    # Testbench file
    if copy_testbench:
        path2testbench = join(getcwd(), f'testbench_verilog/mavg_testbench.v')
        testbench_file = read_template_design_file(path2testbench)
        tb_file = read_template_design_file(testbench_file['func'], params)

        with open(join(path2save, f'mavg_testbench_{module_id_used.lower()}.v'), 'w') as v_handler:
            for line in tb_file:
                v_handler.write(line)
        v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'

    generate_moving_avg_files(16, True, 1, 10, '0',  path2save=path2save)
