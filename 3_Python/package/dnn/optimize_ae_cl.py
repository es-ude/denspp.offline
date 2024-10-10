import numpy as np
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.handler.train_aecl import do_train_ae_classifier


if __name__ == "__main__":
    sets = Config_ML_Pipeline(
        mode_dnn=0,
        do_plot=True,
        do_block=True,
        autoencoder_mode=0,
        autoencoder_feat_size=0,
        autoencoder_noise_std=0.05
    )

    size_hidden_layer = np.arange(1, 20, 3, dtype=int).tolist()

    # --- Iteration
    metrics_runs = dict()
    for hidden_size in size_hidden_layer:
        sets.autoencoder_feat_size = hidden_size
        result = do_train_ae_classifier(sets)

        metrics_runs.update({f"hidden_layer_{hidden_size:02d}": hidden_size, 'results': result})
