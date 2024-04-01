from package.dnn.pytorch_handler import copy_handler_dummy

# TODO: Handler for Training implementieren (Line 7-13)
# TODO: Remove output estimation (getting the information from model)
if __name__ == "__main__":
    # --- Configs
    mode_dnn_train = 0
    mode_cell_bib = 0
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
            # --- Autoencoder
            from src_dnn.train_ae import do_train_ae
            do_train_ae(mode_ae, noise_std_ae, mode_cell_bib, do_plot, do_block)
        case 1:
            # --- Classifier
            from src_dnn.train_cl import do_train_cl
            do_train_cl(num_output, mode_cell_bib, do_plot, do_block)
        case _:
            print("Wrong model! Please select right model!")
