from package.dnn.dnn_handler import dnn_handler

dnn_handler = dnn_handler(
    mode_dnn=4,
    mode_cellbib=2,
    do_plot=True,
    do_block=True
)

# TODO: Remove output estimation (getting the information from model)
if __name__ == "__main__":
    # --- Configs (AE)
    mode_ae = 0
    noise_std_ae = 0.01
    num_output = 5

    # --- Selecting model for train
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- Spike Detection
            from src_dnn.train_sda import dnn_train_sda
            dnn_train_sda(dnn_handler,4)
        case 1:
            # --- Autoencoder (Normal)
            from src_dnn.train_ae import do_train_ae
            do_train_ae(dnn_handler, mode_ae, noise_std_ae)
        case 2:
            # --- Autoencoder + Classifier
            from src_dnn.train_ae_class import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler, num_output, mode_ae, noise_std_ae)
        case 3:
            # --- RGC ON/OFF Classifier
            from src_dnn.train_rgc_class import do_train_rgc_class
            do_train_rgc_class(dnn_handler)
        case 4:
            # --- Neural Decoder (Utah Array)
            from src_dnn.train_decoder_utah import do_train_decoder_utah
            do_train_decoder_utah(dnn_handler, 500)
        case _:
            print("Wrong model! Please select right model!")
