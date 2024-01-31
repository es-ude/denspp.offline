from numpy import median
import matplotlib.pyplot as plt
import csv, os
from package.plotting.plot_dnn import plot_statistic_data, results_training
from package.plotting.plot_metric import plot_loss, prep_confusion
from settings_ai import config_train_class, config_train_ae, config_dataset
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.dataset.autoencoder_class import prepare_training as prepare_training_class
from package.dnn.dataset.autoencoder import prepare_training as prepare_training_ae

from package.dnn.dnn_training import do_train_dnn

do_train_dnn(3, 0.00, 1)
