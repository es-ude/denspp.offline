from package.dnn.dnn_handler import dnn_handler


if __name__ == "__main__":
    dnn_handler = dnn_handler(
        mode_dnn=4,
        mode_cellbib=2,
        do_plot=True,
        do_block=False
    )

    # --- Configs (AE)
    mode_ae = 0
    noise_std_ae = 0.01
    num_hiddenlayer = 5
    num_output = 6

    # --- Selecting model for train
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)"
          "\n===========================================================================================")
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from src_dnn.train_mnist import do_train_cl
            do_train_cl(dnn_handler.do_plot, dnn_handler.do_block)
        case 1:
            # --- MNIST (Autoencoder)
            from src_dnn.train_mnist import do_train_ae
            do_train_ae(dnn_handler.do_plot, dnn_handler.do_block)
        case 2:
            # --- Spike Detection
            from src_dnn.train_sda import dnn_train_sda
            dnn_train_sda(dnn_handler, 4)
        case 3:
            # --- Autoencoder (Normal)
            from src_dnn.train_ae import do_train_ae
            do_train_ae(dnn_handler, mode_ae, noise_std_ae)
        case 4:
            # --- Autoencoder + Classifier
            from src_dnn.train_ae_class import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler, num_hiddenlayer, num_output, mode_ae, noise_std_ae)
        case 5:
            # --- RGC ON/OFF Classifier
            from src_dnn.train_rgc_class import do_train_rgc_class
            do_train_rgc_class(dnn_handler)
        case 6:
            # --- Neural Decoder (Utah Array)
            from src_dnn.train_decoder_utah import do_train_decoder_utah
            do_train_decoder_utah(dnn_handler, 500)
        case _:
            print("Wrong model! Please select right model!")
    print("================================================================"
          "\nFinish!")
