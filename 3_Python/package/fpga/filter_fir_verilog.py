from os import mkdir, getcwd
from os.path import join, isdir
from numpy import log2, ceil
from datetime import datetime
from fxpmath import Fxp

from package.fpga.helper.emulator_filter import filter_stage
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_fir_filter_files(data_bitsize: int, data_bitfrac: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, filter_corner: list,
                              module_id='0', filter_btype='low', filter_ftype='butter', use_ram_coeff=False,
                              do_optimized=False, file_name='filter_fir', path2save='') -> None:
    """Generating Verilog files for FIR filtering on FPGAs/ASICs
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
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    use_dsp_slice_fpga = True
    if do_optimized and filter_order % 2 == 0:
        raise NotImplementedError("Please add an odd number to filter order!")

    # --- Getting the design template
    used_template_name = 'filter_fir_full_template' if not do_optimized else 'filter_fir_half_template'
    path2template = join(getcwd(), f'template_verilog/{used_template_name}.v')
    template_file = read_template_design_file(path2template)

    # --- Reading the filter coefficients
    filter_emulator = filter_stage(filter_order, sampling_rate, filter_corner, False,
                                   ftype=filter_ftype, btype=filter_btype)
    filter_coeff = filter_emulator.get_coeff_quant(data_bitsize, data_bitfrac, data_signed)
    filter_coeff_used = filter_coeff['coeffb'] if not do_optimized else filter_coeff['coeffb'][:int(filter_order/2)+1]

    coeff_data_string = ''
    for idx, coeff in enumerate(filter_coeff_used):
        coeff_data_string += f'\t\tassign coeff_b[{idx:03d}] = {data_bitsize}\'h{coeff.hex()[2:]}; \t//coeff_b[{idx}] = {coeff}\n'

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':     datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'use_int_weights':      '' if use_ram_coeff else '//',
        'use_own_multiplier':   '' if not use_dsp_slice_fpga else '//',
        'device_id':            str(module_id),
        'bitwidth_data':        str(data_bitsize),
        'bitwidth_frac':        str(data_bitfrac),
        'signed_data':          '0' if data_signed else '1',
        'filter_order':         str(filter_order),
        'length_coeff':         str(int(filter_order/2)+1),
        'size_cnt_reg':         str(int(ceil(log2(filter_order)))),
        'filter_type':          f'{filter_btype}, {filter_ftype}',
        'filter_corner':        ', '.join(map(str, filter_corner)),
        'sampling_rate':        f'{sampling_rate}',
        'coeff_data':           coeff_data_string
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

    generate_fir_filter_files(14, 10, True, 21, 1e3, [100],
                              module_id='full', do_optimized=False, path2save=path2save)
    generate_fir_filter_files(14, 10, True, 21, 1e3, [100],
                              module_id='half', do_optimized=True, path2save=path2save)
