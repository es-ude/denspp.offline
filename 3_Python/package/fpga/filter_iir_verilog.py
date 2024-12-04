from os import mkdir, getcwd
from os.path import join, isdir
from datetime import datetime
from fxpmath import Fxp
from shutil import copyfile

from package.fpga.helper.emulator_filter import filter_stage
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_iir_filter_files(data_bitsize: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, filter_corner: list,
                              filter_btype='low', filter_ftype='butter',
                              mode_multiplier=0, use_fast_iir=False, use_ram_coeff=False, weights_bitsize=0,
                              copy_testbench=False,
                              module_id='', file_name='filter_iir', path2save='') -> None:
    """Generating Verilog files for IIR filtering (SOS structure / 2nd filter order) on FPGAs/ASICs
    (Fraction of all weights have data_bitsize-2)
    Args:
        data_bitsize:       Used quantization level for data stream
        data_bitfrac:       Fraction size of used bitwidth
        data_signed:        Decision if data stream in-/output values are signed [otherwise unsigned]
        filter_order:       Order of the filter
        sampling_rate:      Sampling clock of data stream processing
        filter_corner:      List with corner frequency of the used filter
        module_id:          ID of used filter structure
        filter_btype:       Used filter type ['low', 'high', 'bandpass', 'bandstop']
        filter_ftype:       Used filter design ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        use_ram_coeff:      Using a RAM for saving the filter coefficients inside
        mode_multiplier:    Mode of multiplier (0= DSP slice from FPGA, 1= LUT Multiplier, 2= Ext. multiplier)
        weights_bitsize:    Bitsize of all weights [Default: 0 --> data_bitsize]
        copy_testbench:     Copy the template testbench file to output folder
        file_name:          Name of the generated files
        path2save:          Path for saving the verilog output files
    Return:
        None
    """
    module_id_used = module_id if module_id else '0'
    used_bitsize_weights = data_bitsize if weights_bitsize == 0 else weights_bitsize
    if filter_order > 2:
        raise NotImplementedError("Please reduce filter_order to 1 or 2!")
    filter_order_effective = filter_order if filter_btype == 'low' or filter_btype == 'high' else 1

    # --- Getting the design template
    used_template_name = 'filter_iir_onecyc_template' if use_fast_iir else 'filter_iir_fivecyc_template'
    path2template = join(getcwd(), f'template_verilog/{used_template_name}.v')
    template_file = read_template_design_file(path2template)

    # --- Reading the filter coefficients
    filter_emulator = filter_stage(filter_order_effective, sampling_rate, filter_corner, True,
                                   ftype=filter_ftype, btype=filter_btype)
    filter_coeff = filter_emulator.get_coeff_full()
    coeff_string = ''
    for idx, coeff in enumerate(filter_coeff['coeffb']):
        coeff_calc = Fxp(coeff, True, used_bitsize_weights, used_bitsize_weights-2)
        coeff_string += (f'\t\tassign coeff[{idx:01d}] = {used_bitsize_weights}\'h{coeff_calc.hex()[2:]};'
                               f'\t//coeff_b[{idx}] = {coeff_calc.astype(float)}\n')
    for idx, coeff in enumerate(filter_coeff['coeffa'][1:], start=3):
        coeff_calc = Fxp(-coeff, True, used_bitsize_weights, used_bitsize_weights-2)
        coeff_string += (f'\t\tassign coeff[{idx:01d}] = {used_bitsize_weights}\'h{coeff_calc.hex()[2:]};'
                               f'\t//coeff_a[{idx-2}] = {coeff_calc.astype(float)}\n')

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':    datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'use_ram_coeff':        '' if not use_ram_coeff else '//',
        'use_ext_mult':         '' if mode_multiplier == 2 else '//',
        'use_lut_mult':         '' if mode_multiplier == 1 else '//',
        'device_id':            module_id_used.upper(),
        'bitwidth_data':        str(data_bitsize),
        'bitwidth_weights':     str(used_bitsize_weights),
        'signed_data':          '0' if data_signed else '1',
        'filter_type':          f'{filter_btype}, {filter_ftype}',
        'filter_corner':        ', '.join(map(str, filter_corner)),
        'sampling_rate':        f'{sampling_rate}',
        'coeff_data':           coeff_string,
    }

    # --- Write new design to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    if mode_multiplier == 1:
        copyfile(join(getcwd(), f'testbench_verilog/mult_lut_signed_tb.v'),
                 f'{path2save}/mult_lut_signed_testbench.v')
        copyfile('template_verilog/mult_lut_signed.v', f'{path2save}/mult_lut_signed.v')

    # Design file
    imple_file = replace_variables_with_parameters(template_file['func'], params)
    with open(join(path2save, f'{file_name}{module_id_used.lower()}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()

    # Testbench file
    if copy_testbench:
        path2testbench = join(getcwd(), f'testbench_verilog/iir_tb.v')
        testbench_file = read_template_design_file(path2testbench)
        tb_file = replace_variables_with_parameters(testbench_file['func'], params)

        with open(join(path2save, f'iir_testbench_{module_id_used.lower()}.v'), 'w') as v_handler:
            for line in tb_file:
                v_handler.write(line)
        v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'

    generate_iir_filter_files(16, True, 2, 1e3, [100],
                              use_fast_iir=True, module_id='0', path2save=path2save)
    generate_iir_filter_files(16, True, 2, 1e3, [100],
                              use_fast_iir=False, module_id='1', path2save=path2save)
