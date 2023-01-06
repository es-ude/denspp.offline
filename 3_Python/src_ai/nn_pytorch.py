import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split
from src_ai.nnpy_architecture import nn_autoencoder

class NeuralNetwork (nn_autoencoder):
    def __init__(self):
        # --- Properties
        self.os_type = os.name
        self.device = self.__setup()
        self.nn_mode = "Use"

        self.__path2models = "models"
        self.__path2logs = "logs"
        self.__path2fig = "figures"
        # --- Definition of the data
        self.__data_input = None
        self.__data_output = None
        # --- Splitting into training and validation datasets
        self.__train_input = None
        self.__train_output = None
        self.__valid_input = None
        self.__valid_output = None
        # --- Model instanziation
        self.__model_loaded = False

    def initTrain(self, train_size: float, valid_size: float, shuffle: bool, input_size: int, model_name: str):
        nn_autoencoder.__init__(self, input_size)
        self.nn_mode = "Train"

        self.train_size = train_size
        self.valid_size = valid_size
        self.do_shuffle_data = shuffle
        self.__model_name = model_name + "_PyTorch"
        self.model = self.model0

    def __setup(self):
        # TODO: Differenzierung für Betriebssysteme einfügen
        if self.os_type == "nt":
            ostype = "Windows"
        elif self.os_type == "mac":
            ostype = "MAC"
        elif self.os_type == "posix":
            ostype = "Linux"

        # TODO: Einbindung der AMD ROCm für AMD-Support (aktuell kein Paket für Windows)
        device0 = "CUDA" if torch.cuda.is_available() else "CPU"
        if device0 == "CUDA":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("... using PyTorch with", device0, "device on", ostype)
        return device

    def load_data(self, input: np.ndarray, output: np.ndarray, do_norm: bool = False):
        self.__data_input = input
        self.__data_output = output

        # --- Normalization of the data
        if do_norm:
            pass

        # --- Splitting the datasets into training and validation
        (Xin, Yin, Xout, Yout) = train_test_split(
            self.__data_input, self.__data_output,
            train_size=self.train_size,
            test_size=self.valid_size,
            shuffle=self.do_shuffle_data
        )

        # --- Converting data format from NumPy to Torch.Tensor
        self.__train_input = torch.from_numpy(Xin.astype("double"))
        self.__valid_input = torch.from_numpy(Yin.astype("double"))
        self.__train_output = torch.from_numpy(Xout.astype("double"))
        self.__valid_output = torch.from_numpy(Yout.astype("double"))

        # --- Transfering model and data to device
        self.__train_input = self.__train_input
        self.__train_output = self.__train_output

    def train_model(self, batch_size: float, epochs: int, learning_rate: float):
        loss_function = torch.nn.MSELoss()
        # TODO: Implement other optimizer over definition
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )

        print("... start training")
        losses = []
        metric = []
        for epoch in range(epochs):
            inputs = self.__train_input.to(self.device)
            labels = self.__train_output.to(self.device)

            optimizer.zero_grad()

            output = self.model(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)

            print(f"Epcoh {epoch}/{epochs}: Loss = {loss}")

        print("... ANN model is trained")
        pass

    def save_results(self, model_save: bool):
        path2file = os.path.join(self.__path2models, self.__model_name)
        if model_save:
            self.__save_model(self.__path2models, self.__model_name)
        else:
            self.__save_weights(self.__path2models, self.__model_name)

    def print_model(self):
        print("... printing structure and parameters of the neural network")
        print(self.model)

        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def load_results(self):
        pass

    def __save_weights(self, path2model: str, name: str):
        path2file = os.path.join(path2model, name)
        model = models.vgg16(pretrained=True)
        torch.save(model.state_dict(), path2file)

    def __save_model(self, path2model: str, name: str):
        save_file = os.path.join(path2model, name)
        torch.save(self.model, save_file)

    def __load_weights(self, path2model: str):
        model = models.vgg16()
        model.load_state_dict(torch.load(path2model))
        model.eval()

    def __load_model(self, path2model: str, name: str):
        load_model = os.path.join(path2model, name)
        self.model = torch.load(load_model)
