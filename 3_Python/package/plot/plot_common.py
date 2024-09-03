import numpy as np
import os


sel_color = ['k', 'r', 'gray', 'b', 'g', 'y', 'c', 'm']
sel_marker = '.+x_'


def get_plot_color(idx: int) -> 'str':
    """Getting the color string"""
    return sel_color[idx % len(sel_color)]


def cm_to_inch(value: float) -> float:
    """Translation figure size"""
    return value / 2.54


def save_figure(fig, path: str, name: str, formats=('pdf', 'svg')) -> None:
    """Saving figure in given format
    Args:
        fig:        Matplot which will be saved
        path:       Path for saving the figure
        name:       Name of the plot
        formats:    List with data formats for saving the figures
    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)

    path2fig = os.path.join(path, name)
    for idx, form in enumerate(formats):
        fig.savefig(f"{path2fig}.{form}", format=form)


def scale_auto_value(data: np.ndarray | float) -> [float, str]:
    """Getting the scaling value and corresponding string notation for unit scaling in plots
    Args:
        data:   Array or value for calculating the SI scaling value
    Returns:
        Tuple with [0] = scaling value and [1] = SI pre-unit
    """
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
