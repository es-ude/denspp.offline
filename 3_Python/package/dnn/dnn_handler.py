from dataclasses import dataclass


@dataclass(frozen=True)
class dnn_handler:
    """Handling for training depp neural networks"""
    mode_train_dnn: int
    do_plot: bool
    do_block: bool


RecommendedDNNHandler = dnn_handler(
    mode_train_dnn=0,
    do_plot=True,
    do_block=True
)
