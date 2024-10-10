import os
from datetime import datetime
import numpy as np
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import (Config_Dataset, DefaultSettingsDataset,
                                           Config_PyTorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE)
from package.plot.plot_metric import plot_confusion, calculate_class_accuracy
from package.plot.plot_dnn import plot_statistic_data

from package.dnn.template.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.template.dataset.autoencoder_class import prepare_training as get_dataset_class
from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
from package.dnn.pytorch.classifier import train_nn as train_classifier
import package.dnn.template.models.autoencoder_cnn as models_ae
import package.dnn.template.models.autoencoder_class as models_class


# Configuration for the dataset
config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../../2_Data/00_Merged_Datasets',
    data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',
    # --- Data Augmentation
    augmentation_do=False,
    augmentation_num=0,
    add_noise_cluster=False,
    # --- Data Normalization
    normalization_do=True,
    normalization_mode='CPU',
    normalization_method='minmax',
    normalization_setting='bipolar',
    # --- Dataset Reduction
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=5_000,
    exclude_cluster=[],
    reduce_positions_per_sample=[]
)


def do_train_ae_classifier(settings: Config_ML_Pipeline,
                           num_feature_layer: int, num_output_cl: int,
                           mode_ae=0, noise_std=0.05, path2save_base='',
                           num_epochs=5) -> dict:
    """Training routine for Autoencoders and Classification after Encoder"""
    metric_run = dict()

    # --- Definition of settings
    config_train_ae = Config_PyTorch(
        model=models_ae.cnn_ae_v4(32, num_feature_layer),
        loss='MSE',
        loss_fn=nn.MSELoss(),
        optimizer='Adam',
        num_kfold=1,
        num_epochs=num_epochs,  # Using the same variable
        batch_size=256,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )
    config_train_cl = Config_PyTorch(
        model=models_class.classifier_ae_v1(num_feature_layer, num_output_cl),
        loss='Cross Entropy',
        loss_fn=nn.CrossEntropyLoss(),
        optimizer='Adam',
        num_kfold=1,
        num_epochs=num_epochs,  # Defined
        batch_size=256,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )

    # --- Loading YAML files


    # ----------- Step #1: TRAINING AUTOENCODER
    dataset = get_dataset_ae(settings=config_data, mode_train_ae=mode_ae, noise_std=noise_std, do_classification=False)
    trainhandler = train_autoencoder(config_train=config_train_ae, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)

    # Create a subdirectory for Autoencoder and pass it to do_training
    path2save_ae = os.path.join(path2save_base, f'Iteration_{num_feature_layer}', 'AE')
    if not os.path.exists(path2save_ae):
        os.makedirs(path2save_ae)
    loss_ae = trainhandler.do_training(path2save=path2save_ae)[-1][0]

    metric_run.update({"train_loss_ae": loss_ae[-1][0]})
    metric_run.update({"valid_loss_ae": loss_ae[-1][1]})

    path2model = trainhandler.get_saving_path()

    # ----------- Step #2: TRAINING CLASSIFIER
    dataset = get_dataset_class(settings=config_data, path2model=path2model)
    num_output = dataset.__frames_me.shape[0]
    trainhandler = train_classifier(config_train=config_train_cl, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)

    # Create a subdirectory for Classifier and pass it to do_training
    path2save_cl = os.path.join(path2save_base, f'Iteration_{num_feature_layer}', 'Classifier')
    if not os.path.exists(path2save_cl):
        os.makedirs(path2save_cl)
    training_results = trainhandler.do_training(path2save=path2save_cl)[-1]  # Training results from the last epoch
    acc_class_train = training_results[0][-1]  # Training accuracy
    acc_class_valid = training_results[1][-1]  # Validation accuracy
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training(num_output)

    # ---- Store Accuracy, F1-Score, and Class Accuracies ----
    class_accuracies = calculate_class_accuracy(data_result['valid_clus'], data_result['yclus'])

    precision, recall, f1_score, _ = precision_recall_fscore_support(data_result['valid_clus'], data_result['yclus'],
                                                                     average='macro')
    metric_run.update({
        "accuracy_class": [acc_class_train, acc_class_valid],  # Store both training and validation accuracy
        "f1_score": f1_score,  # Store F1 score
        "class_accuracies": class_accuracies,
        "path2save": logsdir
    })

    if settings.do_plot:
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=logsdir,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=settings.do_block)

    del dataset, trainhandler

    return metric_run


if __name__ == "__main__":
    from package.dnn.dnn_handler import Config_ML_Pipeline
    import matplotlib.pyplot as plt

    dnn_handler = Config_ML_Pipeline(
        mode_dnn=4,
        mode_cellbib=2,
        do_plot=False,
        do_block=True
    )

    # --- Create the parent folder for saving configurations
    data_file_name = config_data.data_file_name
    parent_folder_name = f"Evaluation config.-DATAFILE-{data_file_name}"

    # Replace any invalid characters in folder_name
    parent_folder_name = parent_folder_name.replace(':', '_').replace(' ', '_').replace('.', '_')

    path2save_base = os.path.join('runs', parent_folder_name)

    # Create the base directory if it doesn't exist
    if not os.path.exists(path2save_base):
        os.makedirs(path2save_base)

    # Get the AE and Classifier model versions
    ae_model_function = models_ae.cnn_ae_v4
    classifier_model_function = models_class.classifier_ae_v1

    ae_version = ae_model_function.__name__
    classifier_version = classifier_model_function.__name__

    metrics_runs = dict()

    # List of epoch numbers (adjustable)
    epoch_numbers = [5, 30, 100] #+ list(range(60, 101, 10))

    for num_epochs in epoch_numbers:
        if num_epochs <= 50:
            max_hidden_layer_neurons = 20
        else:
            max_hidden_layer_neurons = 20  # Adjust if needed

        size_hidden_layer = np.arange(1, max_hidden_layer_neurons + 1, 1, dtype=int).tolist()

        # Create a folder for this epoch number
        epoch_folder_name = f"Epochs_{num_epochs}"
        epoch_folder_path = os.path.join(path2save_base, epoch_folder_name)
        if not os.path.exists(epoch_folder_path):
            os.makedirs(epoch_folder_path)

        # Set path2save_base_epoch for this epoch number
        path2save_base_epoch = epoch_folder_path  # Updated path2save_base for this epoch

        # Initialize lists to store values for plotting for this epoch number
        hidden_layer_sizes_for_plot = []
        loss_train_values_for_plot = []
        loss_valid_values_for_plot = []
        class_accuracies_for_plot = []
        accuracies_for_plot = []
        f1_scores_for_plot = []

        for idx, hidden_size in enumerate(size_hidden_layer):
            result = do_train_ae_classifier(
                dnn_handler,
                hidden_size,
                4,
                path2save_base=path2save_base_epoch,  # Use the updated path
                num_epochs=num_epochs
            )

            result.update({"Size_Hiddenlayer": hidden_size, "NumEpochs": num_epochs})
            metrics_runs.update({f"Run_Epochs_{num_epochs}_Hidden_{hidden_size}": result})

            hidden_layer_sizes_for_plot.append(hidden_size)
            loss_train_values_for_plot.append(result["train_loss_ae"])
            loss_valid_values_for_plot.append(result["valid_loss_ae"])
            class_accuracies_for_plot.append(result["class_accuracies"])
            # Stores validation accuracy. It can be changed to [1] if training accuracy is also needed.
            accuracies_for_plot.append(result["accuracy_class"][1][0])
            f1_scores_for_plot.append(result["f1_score"])

        # After processing all hidden sizes for this num_epochs, save plots

        # --- Write parameters to a .txt file in the epoch folder
        txt_file_name = f"NumEP={num_epochs}_HidNeu={max_hidden_layer_neurons}.txt"
        info_file_path = os.path.join(epoch_folder_path, txt_file_name)
        with open(info_file_path, 'w') as info_file:
            info_file.write('--- Evaluation Parameters ---\n')
            info_file.write(f'Maximum number of hidden layer neurons: {max_hidden_layer_neurons}\n')
            info_file.write(f'Number of epochs for Autoencoder and Classifier: {num_epochs}\n')

        # --- Create a 'Plots' folder inside the epoch folder
        plots_folder = os.path.join(epoch_folder_path, 'Plots')
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Convert class accuracies to numpy array for easier handling
        class_accuracies_for_plot = np.array(class_accuracies_for_plot)

        # Plotting Training and Validation Loss vs. Number of Neurons in Hidden Layer
        plt.figure(figsize=(10, 6))
        plt.plot(hidden_layer_sizes_for_plot, loss_train_values_for_plot, marker='o', label='Training Loss')
        plt.plot(hidden_layer_sizes_for_plot, loss_valid_values_for_plot, marker='o', label='Validation Loss')
        plt.xlabel('Number of Neurons in Hidden Layer')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss vs. Hidden Layer Size (Epochs={num_epochs})')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save the plot to the 'Plots' folder
        plt.savefig(os.path.join(plots_folder, f'Loss_Epochs_{num_epochs}.png'))
        plt.close()

        # Plotting Validation Overall Accuracy vs. Number of Neurons in Hidden Layer
        plt.figure(figsize=(10, 6))
        plt.plot(hidden_layer_sizes_for_plot, accuracies_for_plot, marker='o', label='Validation Accuracy')
        plt.xlabel('Number of Neurons in Hidden Layer')
        plt.ylabel('Overall Accuracy')
        plt.title(f'Validation Accuracy vs. Hidden Layer Size (Epochs={num_epochs})')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save the plot to the 'Plots' folder
        plt.savefig(os.path.join(plots_folder, f'Validation_Accuracy_Epochs_{num_epochs}.png'))
        plt.close()

        # Plotting F1 Scores vs. Number of Neurons in Hidden Layer
        plt.figure(figsize=(10, 6))
        plt.plot(hidden_layer_sizes_for_plot, f1_scores_for_plot, marker='o', label='F1 Score')
        plt.xlabel('Number of Neurons in Hidden Layer')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs. Hidden Layer Size (Epochs={num_epochs})')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save the plot to the 'Plots' folder
        plt.savefig(os.path.join(plots_folder, f'F1_Score_Epochs_{num_epochs}.png'))
        plt.close()

        # Plotting Class Accuracies
        num_classes = class_accuracies_for_plot.shape[1]

        plt.figure(figsize=(10, 6))
        for class_idx in range(num_classes):
            plt.plot(hidden_layer_sizes_for_plot, class_accuracies_for_plot[:, class_idx], marker='o',
                     label=f'Class {class_idx}')
        plt.xlabel('Number of Neurons in Hidden Layer')
        plt.ylabel('Class Accuracy')
        plt.title(f'Class Accuracy vs. Hidden Layer Size (Epochs={num_epochs})')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save the plot to the 'Plots' folder
        plt.savefig(os.path.join(plots_folder, f'Class_Accuracy_Epochs_{num_epochs}.png'))
        plt.close()

    # --- After the loop over epoch numbers, get the timestamp of the last configuration data
    last_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Rename the parent folder to include the last timestamp
    final_parent_folder_name = f"Evaluation config.-{last_timestamp}-{data_file_name}"
    final_parent_folder_name = final_parent_folder_name.replace(':', '_').replace(' ', '_').replace('.', '_')
    final_path2save_base = os.path.join('runs', final_parent_folder_name)

    os.rename(path2save_base, final_path2save_base)

    # --- Write AE and Classifier versions to a text file in the evaluation folder
    config_data_file_name = 'AE & Classifier Versions.txt'
    config_data_file_path = os.path.join(final_path2save_base, config_data_file_name)
    with open(config_data_file_path, 'w') as config_file:
        config_file.write('--- Autoencoder and Classifier Versions ---\n')
        config_file.write(f'Autoencoder model version: {ae_version}\n')
        config_file.write(f'Classifier model version: {classifier_version}\n')
