from package.dnn.pytorch_handler import copy_handler_dummy


class dnn_handler:
    """Handling for training depp neural networks"""
    mode_train_dnn: int
    mode_cell_bib: int
    do_plot: bool
    do_block: bool

    def __init__(self, mode_dnn: int, mode_cellbib: int, do_plot: bool, do_block=True, do_init=True):
        self.mode_train_dnn = mode_dnn
        self.mode_cell_bib = mode_cellbib
        self.do_plot = do_plot
        self.do_block = do_block

        # --- Checking if templates are available
        if do_init:
            copy_handler_dummy()
