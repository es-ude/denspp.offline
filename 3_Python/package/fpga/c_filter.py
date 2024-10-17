from os import mkdir
from os.path import join, isdir
from datetime import datetime
import numpy as np
from package.fpga.signal_type import generation_sinusoidal_waveform
from package.fpga.emulator_filter import filter_stage


def get_embedded_datatype(bits_used: int, out_signed=True) -> str:
    """Determine the C datatype for processing data
    Args:
        bits_used:      Integer with bit value
        out_signed:     Boolean if datatype is signed or unsigned
    Return:
        String with datatype in C
    """
    # Define datatyp used in embedded device
    bit_compare = np.array((8, 16, 32, 64))
    used_bitval = np.argwhere((bit_compare / bits_used) - 1 >= 0).flatten()[0]
    return ('' if out_signed else 'u') + 'int' + f'{bit_compare[used_bitval]}' + f'_t'


def generate_iir_filter_c(data_bitsize: int, data_signed: bool,
                          filter_order: int, fs: float, filter_corner: float, filter_type='low',
                          do_optimized=False, file_name='filter_iir', path2save='') -> None:
    """Generating C files for IIR filtering on microcontroller
    Args:
        data_bitsize:   Used quantization level for data stream
        data_signed:    Decision if LUT values are signed [otherwise unsigned]
        filter_order:   Order of the filter
        fs:             Sampling clock
        filter_corner:  Corner frequency of the used filter
        filter_type:    Type of the filter used ['low', 'high', 'bandpass', 'bandstop', 'notch', 'all']
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        file_name:      Name of the generated files
        path2save:      Path for saving the verilog output files
    Return:
        None
    """
    filter_emulator = filter_stage(N=filter_order, fs=fs, f_filter=filter_corner,
                                   use_iir_filter=True, btype=filter_type)
    coeff_irr = filter_emulator.get_coeff_full()

    # --- Write lines to file
    # Checking if path is available
    if path2save and not isdir(path2save):
        mkdir(path2save)

    # --- Write lines to file
    """print('... create C file for handling data (*.c)')
    with open(join(path2save, f'{file_name}.c'), 'w') as v_handler:
        for line in file_content:
            v_handler.write(line + '\n')
        list2write = function_call0 if do_optimized else function_call1
        for line in list2write:
            v_handler.write(line + '\n')
    v_handler.close()
    """


if __name__ == '__main__':
    path2save = '../../runs'

    print("\nCreating the C files\n====================================")
    generate_iir_filter_c(14, True, 1, 1e3, 100,
                          do_optimized=False, path2save=path2save)
