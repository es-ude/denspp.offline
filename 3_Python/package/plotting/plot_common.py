from os.path import join


def cm_to_inch(value):
    """Translation figure size"""
    return value / 2.54


def save_figure(fig, path: str, name: str, format=['pdf', 'svg']):
    """Saving figure in given format"""
    path2fig = join(path, name)

    for idx, form in enumerate(format):
        file_name = path2fig + '.' + form
        fig.savefig(file_name, format=form)
