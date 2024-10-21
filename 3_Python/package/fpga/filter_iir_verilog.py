from os import mkdir, getcwd
from os.path import join, isdir
from numpy import log2, ceil
from datetime import datetime
from fxpmath import Fxp

from package.fpga.helper.emulator_filter import filter_stage
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_iir_filter_files(data_bitsize: int, data_bitfrac: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, filter_corner: list,
                              module_id='0', filter_btype='low', filter_ftype='butter',
                              use_fast_iir=False, use_ram_coeff=False, file_name='filter_iir', path2save='') -> None:
    """Generating Verilog files for IIR filtering (SOS structure / 2nd filter order) on FPGAs/ASICs
    Args:
        data_bitsize:   Used quantization level for data stream
        data_bitfrac:   Fraction size of used bitwidth
        data_signed:    Decision if data stream in-/output values are signed [otherwise unsigned]
        filter_order:   Order of the filter
        sampling_rate:  Sampling clock of data stream processing
        filter_corner:  List with corner frequency of the used filter
        module_id:      ID of used filter structure
        filter_btype:   Used filter type ['low', 'high', 'bandpass', 'bandstop']
        filter_ftype:   Used filter design ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        use_ram_coeff:  Using a RAM for saving the filter coefficients inside
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    use_dsp_slice_fpga = True
    if filter_order > 2:
        raise NotImplementedError("Please reduce filter_order to 1 or 2!")

    # --- Getting the design template
    used_template_name = 'filter_iir_onecyc_template' if use_fast_iir else 'filter_iir_fivecyc_template'
    path2template = join(getcwd(), f'template_verilog/{used_template_name}.v')
    template_file = read_template_design_file(path2template)

    # --- Reading the filter coefficients
    filter_emulator = filter_stage(filter_order, sampling_rate, filter_corner, True,
                                   ftype=filter_ftype, btype=filter_btype)
    filter_coeff = filter_emulator.get_coeff_full()
    coeffa_data_string = ''
    for idx, coeff in enumerate(filter_coeff['coeffa']):
        coeff_calc = Fxp(-coeff, data_signed, data_bitsize, data_bitfrac)
        coeffa_data_string += (f'\t\tassign coeff_a[{idx:01d}] = {data_bitsize}\'h{coeff_calc.hex()[2:]};'
                               f'\t//coeff_a[{idx}] = {coeff_calc.astype(float)}\n')

    coeffb_data_string = ''
    for idx, coeff in enumerate(filter_coeff['coeffb']):
        coeff_calc = Fxp(coeff, data_signed, data_bitsize, data_bitfrac)
        coeffb_data_string += (f'\t\tassign coeff_b[{idx:01d}] = {data_bitsize}\'h{coeff.hex()[2:]};'
                               f'\t//coeff_b[{idx}] = {coeff_calc.astype(float)}\n')

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':    datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'use_ram_coeff':        '' if use_ram_coeff else '//',
        'use_own_mult':         '' if not use_dsp_slice_fpga else '//',
        'device_id':            str(module_id),
        'bitwidth_data':        str(data_bitsize),
        'bitwidth_frac':        str(data_bitfrac),
        'signed_data':          '0' if data_signed else '1',
        'filter_order':         str(filter_order),
        'size_cnt_reg':         str(int(ceil(log2(filter_order)))),
        'filter_type':          f'{filter_btype}, {filter_ftype}',
        'filter_corner':        ', '.join(map(str, filter_corner)),
        'sampling_rate':        f'{sampling_rate}',
        'coeffa_data':          coeffa_data_string,
        'coeffb_data':          coeffb_data_string
    }
    imple_file = replace_variables_with_parameters(template_file['func'], params)

    # --- Write new design to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    with open(join(path2save, f'{file_name}_{module_id}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'

    generate_iir_filter_files(14, 10, True, 2, 1e3, [100],
                              use_fast_iir=True, module_id='0', path2save=path2save)
    generate_iir_filter_files(14, 10, True, 2, 1e3, [100],
                              use_fast_iir=False, module_id='1', path2save=path2save)
