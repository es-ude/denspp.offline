from package.dnn.pytorch_handler import copy_handler_dummy

# TODO: Handler for Training implementieren (Line 7-13)
# TODO: Remove output estimation (getting the information from model)
if __name__ == "__main__":
    # --- Configs
    mode_dnn_train = 3
    mode_cell_bib = 2
    mode_ae = 0
    noise_std_ae = 0.01
    num_output = 5
    do_plot = True
    do_block = True

    # --- Generate templates
    copy_handler_dummy()
    # --- Selecting model for train
    match mode_dnn_train:
        case 0:
            # --- Spike Detection
            from src_dnn.train_sda import dnn_train_sda
            dnn_train_sda(4, do_plot, do_block)
        case 1:
            # --- Autoencoder (Normal)
            from src_dnn.train_ae import do_train_ae
            do_train_ae(mode_ae, noise_std_ae, mode_cell_bib, do_plot, do_block)
        case 2:
            # --- Autoencoder + Classifier
            from src_dnn.train_ae_class import do_train_ae_classifier
            do_train_ae_classifier(num_output, mode_ae, noise_std_ae, mode_cell_bib, do_plot, do_block)
        case 3:
            # --- RGC ON/OFF Classifier
            from src_dnn.train_rgc_class import do_train_rgc_class
            do_train_rgc_class(mode_cell_bib, do_plot, do_block)
        case 4:
            # --- Neural Decoder (Utah Array)
            from src_dnn.train_decoder_utah import do_train_decoder_utah
            do_train_decoder_utah(500, do_plot, do_block)
        case _:
            print("Wrong model! Please select right model!")
