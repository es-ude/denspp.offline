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
