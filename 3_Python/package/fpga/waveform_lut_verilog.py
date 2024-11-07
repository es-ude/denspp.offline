from os import mkdir, getcwd
from os.path import join, isdir
from datetime import datetime
import numpy as np
from package.fpga.helper.signal_type import generation_sinusoidal_waveform
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_waveform_lut(lut_bitsize: int, lut_signed: bool,
                          f_sys: float, f_rpt: float, f_wvf: float,
                          module_id='', copy_testbench=False, do_optimized=False,
                          file_name='waveform_lut', path2save='') -> None:
    """Generating Verilog file for generating waveform using LUTs
    Args:
        lut_bitsize:    Used quantization level for generating sinusoidal waveform LUT
        lut_signed:     Decision if LUT values are signed [otherwise unsigned]
        f_sys:          System clock used in FPGA design
        f_rpt:          Frequency of the timer interrupt
        f_wvf:          Target frequency of the sinusoidal waveform at output
        module_id:      ID of generated waveform LUT generator
        copy_testbench: Copy the template testbench file to output folder
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    do_cnt_external = False
    do_read_external = False
    do_trgg_external = False

    module_id_used = module_id if module_id else '0'
    # --- Getting the design template
    use_template_part = 'full' if not do_optimized else 'opt'
    path2template = join(getcwd(), f'template_verilog/waveform_lut_{use_template_part}_template.v')
    template_file = read_template_design_file(path2template)

    # --- Definition of used parameters and replacement
    lut_data = generation_sinusoidal_waveform(lut_bitsize, f_rpt, f_wvf,
                                              do_signed=lut_signed, do_optimized=do_optimized)
    lut_bitsize_array = lut_bitsize if not do_optimized else lut_bitsize - 1
    lut_string_array = ''
    lut_string_stream = ''
    for idx, value in enumerate(lut_data):
        lut_string_array += f'\t\tassign lut_ram[{idx}] = {lut_bitsize_array}\'d{value};\n'
    for value in reversed(lut_data):
        lut_string_stream += f'{lut_bitsize_array}\'d{value}, '
    lut_string_stream = lut_string_stream[:-2]

    params = {
        'num_sinelut':              str(lut_data.size),
        'bitsize_lut':              str(lut_bitsize),
        'device_id':                module_id_used,
        'wait_cycles':              str(int(np.ceil(f_sys / f_rpt))),
        'wait_cnt_width':           str(int(np.ceil(np.log2(np.ceil(f_sys / f_rpt))))),
        'signed_type':              'signed' if lut_signed and not do_optimized else '',
        'lut_data_array':           lut_string_array,
        'lut_data_stream':          lut_string_stream,
        'do_unsigned_call':         '//' if lut_signed else '',
        'do_signed_call':           '//' if not lut_signed else '',
        'period_cycles':            str(int(np.ceil(f_sys / f_wvf))),
        'do_read_lut_external':     '//' if not do_read_external else '',
        'do_cnt_external':          '//' if not do_cnt_external else '',
        'do_trgg_external':         '//' if not do_trgg_external else '',
        'date_copy_created':        datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    }

    # --- Generating files
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    # Design file
    imple_file = replace_variables_with_parameters(template_file['func'], params)
    with open(join(path2save, f'{file_name}{module_id_used.lower()}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()

    # Testbench file
    if copy_testbench:
        path2testbench = join(getcwd(), f'testbench_verilog/waveform_lut_tb.v')
        testbench_file = read_template_design_file(path2testbench)
        tb_file = replace_variables_with_parameters(testbench_file['func'], params)

        with open(join(path2save, f'waveform_lut_{module_id_used.lower()}_tb.v'), 'w') as v_handler:
            for line in tb_file:
                v_handler.write(line)
        v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'
    n_bit = 8
    f_sys = 100e6
    f_rpt = 200e3
    f_sine = 10e3

    generate_waveform_lut(n_bit, False, f_sys, f_rpt, f_sine, module_id='0',
                          do_optimized=False, copy_testbench=True, path2save=path2save)
    generate_waveform_lut(n_bit, True, f_sys, f_rpt, f_sine, module_id='1',
                          do_optimized=True, copy_testbench=False, path2save=path2save)
