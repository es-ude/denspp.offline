import numpy as np
from os.path import join
import matplotlib.pyplot as plt


def _cm_to_inch(value):
    """Translation figure size"""
    return value / 2.54


def _save_figure(fig, path: str, name: str, formats=('pdf', 'svg')):
    """Saving figure in given format"""
    path2fig = join(path, name)

    for idx, form in enumerate(formats):
        fig.savefig(f"{path2fig}.{form}", format=form)


def _show_plots(block=True) -> None:
    """Showing plots and blocking system if required"""
    plt.show(block=block)


def _close_plots() -> None:
    """Closing all opened plots"""
    plt.close('all')


def _scale_auto_value(data: np.ndarray | float) -> [float, str]:
    """Getting the scaling value and corresponding string notation for unit scaling in plots"""
    if isinstance(data, np.ndarray):
        value = np.max(np.abs(data))
    else:
        value = data

    str_value = str(value).split('.')
    digit = 0
    if 'e' not in str_value[1]:
        if not str_value[0] == '0':
            # --- Bigger Representation
            sys = -np.floor(len(str_value[0]) / 3)
        else:
            # --- Smaller Representation
            for digit, val in enumerate(str_value[1], start=1):
                if '0' not in val:
                    break
            sys = np.ceil(digit / 3)
    else:
        val = int(str_value[1].split('e')[-1])
        sys = -np.floor(abs(val) / 3) if np.sign(val) == 1 else np.ceil(abs(val) / 3)

    scale = 10 ** (sys * 3)
    match sys:
        case -4:
            units = 'T'
        case -3:
            units = 'G'
        case -2:
            units = 'M'
        case -1:
            units = 'k'
        case 0:
            units = ''
        case 1:
            units = 'm'
        case 2:
            units = 'µ'
        case 3:
            units = 'n'
        case 4:
            units = 'p'
        case 5:
            units = 'f'
        case _:
            units = 'f'

    return scale, units
