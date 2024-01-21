from package.dnn.dnn_training import do_train_dnn, check_settings_file

if __name__ == "__main__":
    check_settings_file()

    do_train_dnn(
        mode_train=4,
        noise_std_ae=0.0,
        mode_cell_bib=1,
        only_plot_results=False
    )
