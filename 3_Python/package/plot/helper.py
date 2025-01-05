import os
import numpy as np
import matplotlib.pyplot as plt


def get_textsize_paper() -> int:
    """Getting the fontsize (best practice) for publishing in papers"""
    return 14


def get_plot_color_inactive() -> str:
    """Getting the color for plotting non-spike activity in transient plots"""
    return '#929591'


def get_plot_color(idx: int) -> str:
    """Getting the color string"""
    sel_color = ['k', 'r', 'b', 'g', 'y', 'c', 'm', 'gray']
    return sel_color[idx % len(sel_color)]


def get_marker(idx: int) -> str:
    """Getting the marker for plotting"""
    sel_marker = '.+x_'
    return sel_marker[idx % len(sel_marker)]


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


def _get_median(parameter: list) -> float:
    """Calculating the median of list input"""
    param = np.zeros(shape=(len(parameter), ), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.median(val)
    return float(np.median(param))


def _get_mean(parameter: list) -> float:
    """Calculating the mean of list input"""
    param = np.zeros(shape=(len(parameter), ), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.mean(val)

    return float(np.mean(param))


def _autoscale(ax=None, axis='y', margin=0.1):
    """Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified."""

    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x,y = _get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = _calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin*(newhigh - newlow)
    setlim(newlow-margin, newhigh+margin)


def _calculate_new_limit(fixed, dependent, limit):
    """Calculates the min/max of the dependent axis given a fixed axis with limits"""
    if len(fixed) > 2:
        mask = (fixed > limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high


def _get_xy(artist):
    """Gets the xy coordinates of a given artist"""
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y
