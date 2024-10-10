import numpy as np
from package.dnn.dnn_handler import dnn_handler
from package.dnn.template.handler.train_aecl import do_train_ae_classifier


if __name__ == "__main__":
    sets = dnn_handler(
        mode_dnn=0,
        do_plot=True,
        do_block=True
    )
    size_hidden_layer = np.arange(1, 20, 3, dtype=int).tolist()

    # --- Iteration
    metrics_runs = dict()
    for idx, hidden_size in enumerate(size_hidden_layer):
        result = do_train_ae_classifier(sets, hidden_size, 4)
        metrics_runs.update({f"hidden_layer_{hidden_size:02d}": hidden_size, 'results': result})
