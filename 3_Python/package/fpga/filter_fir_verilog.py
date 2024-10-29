from os import mkdir, getcwd
from os.path import join, isdir
from numpy import log2, ceil
from datetime import datetime

from package.fpga.helper.emulator_filter import filter_stage
from package.fpga.helper.translater import replace_variables_with_parameters, read_template_design_file


def generate_fir_filter_files(data_bitsize: int, data_signed: bool,
                              filter_order: int, sampling_rate: float, filter_corner: list,
                              module_id='0', filter_btype='low', filter_ftype='butter', use_ext_coeff=False,
                              do_optimized=False, weights_bitsize=0, file_name='filter_fir', path2save='') -> None:
    """Generating Verilog files for FIR filtering on FPGAs/ASICs
    (Fraction of all weights have data_bitsize-2)
    Args:
        data_bitsize:       Used quantization level for data stream
        data_signed:        Decision if data stream in-/output values are signed [otherwise unsigned]
        filter_order:       Order of the filter
        sampling_rate:      Sampling clock of data stream processing
        filter_corner:      List with corner frequency of the used filter
        module_id:          ID of used filter structure
        filter_btype:       Used filter type ['low', 'high', 'bandpass', 'bandstop']
        filter_ftype:       Used filter design ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        use_ext_coeff:      Using an external RAM for saving the filter coefficients inside
        do_optimized:       Decision if LUT resources should be minimized [only quarter and mirroring]
        weights_bitsize:    Bitsize of all weights [Default: 0 --> data_bitsize]
        file_name:          Name of the generated files
        path2save:          Path for saving the verilog output files
    Return:
        None
    """
    use_dsp_slice_fpga = True
    use_ext_mult = False

    if do_optimized and filter_order % 2 == 0:
        raise NotImplementedError("Please add an odd number to filter order!")

    # --- Getting the design template
    used_template_name = 'filter_fir_full_template' if not do_optimized else 'filter_fir_half_template'
    path2template = join(getcwd(), f'template_verilog/{used_template_name}.v')
    template_file = read_template_design_file(path2template)

    # --- Reading the filter coefficients
    filter_emulator = filter_stage(filter_order, sampling_rate, filter_corner, False,
                                   ftype=filter_ftype, btype=filter_btype)
    filter_coeff = filter_emulator.get_coeff_quant(data_bitsize, data_bitsize, True)
    filter_coeff_used = filter_coeff['coeffb'] if not do_optimized else filter_coeff['coeffb'][:int(filter_order/2)+1]

    coeff_data_string = ''
    for idx, coeff in enumerate(filter_coeff_used):
        coeff_data_string += f'\t\tassign coeff_b[{idx:03d}] = {data_bitsize}\'h{coeff.hex()[2:]}; \t//coeff_b[{idx}] = {coeff}\n'

    # --- Definition of used parameters and replacement
    params = {
        'date_copy_created':     datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'use_int_weights':      '' if not use_ext_coeff else '//',
        'use_ext_mult':         '' if use_ext_mult else '//',
        'use_lut_mult':         '' if not use_dsp_slice_fpga else '//',
        'device_id':            str(module_id),
        'bitwidth_data':        str(data_bitsize),
        'bitwidth_weights':     str(data_bitsize) if weights_bitsize == 0 else str(weights_bitsize),
        'signed_data':          '0' if data_signed else '1',
        'filter_order':         str(filter_order) if not do_optimized else str(int(filter_order/2)+1),
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

    with open(join(path2save, f'{file_name}_{module_id.lower()}.v'), 'w') as v_handler:
        for line in imple_file:
            v_handler.write(line)
    v_handler.close()


if __name__ == '__main__':
    path2save = '../../runs'

    generate_fir_filter_files(16, False, 21, 1e3, [100],
                              module_id='FULL', do_optimized=False, path2save=path2save)
    generate_fir_filter_files(16, False, 21, 1e3, [100],
                              module_id='HALF', do_optimized=True, path2save=path2save)
