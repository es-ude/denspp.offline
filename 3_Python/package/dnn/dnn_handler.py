import dataclasses


@dataclasses.dataclass
class dnn_handler:
    """Handling for training depp neural networks"""
    mode_train_dnn = 0
    mode_cell_bib = 0
    do_plot = True
    do_block = True
