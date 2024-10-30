from os import mkdir, getcwd
from os.path import join, isdir
from datetime import datetime
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_fir_allpass_files(data_bitsize: int, t_dly: float, sampling_rate: float,
                               module_id='', copy_testbench=False, file_name='filter_fir_all', path2save='') -> None:
    """Generating Verilog files for FIR all-pass filtering (ideal) on FPGAs/ASICs
    Args:
        data_bitsize:       Used quantization level for data stream
        t_dly:              Delay time in seconds
        sampling_rate:      Sampling clock of data stream processing
        module_id:          ID of used filter structure
        copy_testbench:     Copy the template testbench file to output folder
        file_name:          Name of the generated files
        path2save:          Path for saving the verilog output files
    Return:
        None
    """
    module_id_used = module_id if module_id else '0'
    # --- Getting the design template
    used_template_name = 'filter_fir_allpass_template'
    path2template = join(getcwd(), f'template_verilog/{used_template_name}.v')
    template_file = read_template_design_file(path2template)

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':     datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'device_id':            module_id_used.upper(),
        'bitwidth_data':        str(data_bitsize),
        'filter_order':         str(int(t_dly * sampling_rate)),
        'sampling_rate':        f'{sampling_rate}'
    }

    # --- Write new design to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    # Design File
    imple_file = replace_variables_with_parameters(template_file['func'], params)
    with open(join(path2save, f'{file_name}{module_id_used.lower()}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()

    # Testbench file
    if copy_testbench:
        path2testbench = join(getcwd(), f'testbench_verilog/fir_delay_testbench.v')
        testbench_file = read_template_design_file(path2testbench)
        tb_file = read_template_design_file(testbench_file['func'], params)

        with open(join(path2save, f'fir_delay_testbench{module_id_used.lower()}.v'), 'w') as v_handler:
            for line in tb_file:
                v_handler.write(line)
        v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'
    generate_fir_allpass_files(16, 10e-3, 1e3, path2save=path2save)
