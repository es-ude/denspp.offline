from package.dnn.dnn_training import do_train_dnn

if __name__ == "__main__":
    do_train_dnn(
        mode_train=0,
        noise_std_ae=0.05,
        mode_cell_bib=0,
        only_plot_results=False
    )
