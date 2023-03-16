import os, time
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from torchsummary import summary
from sklearn.model_selection import train_test_split

class NeuralNetwork():
    def __init__(self):
        # --- Properties (general)
        self.os_type = os.name
        self.device = self.__setup()
        self.__path2models = "models"
        self.__path2logs = "logs"
        self.__path2fig = "figures"
        # --- Properties (Model)
        self.model = None
        self.model_name = None
        self.model_input_size = 0
        self.__model_loaded = False
        self.__model_trained = False

        # --- Definition of the data
        self.__data_input = None
        self.__data_output = None
        # --- Splitting into training and validation datasets
        self.__train_input = None
        self.__train_output = None
        self.__valid_input = None
        self.__valid_output = None


    def defineModel(self, model, input_size: int):
        self.model = model
        self.model_name = model.mname
        self.model_input_size = input_size
        self.__model_loaded = True

    def initTrain(self, train_size: float, valid_size: float, shuffle: bool, name_addon: str):
        self.__model_name = self.model_name + name_addon + "_PyTorch"

        self.train_size = train_size
        self.valid_size = valid_size
        self.do_shuffle_data = shuffle

    def initPredict(self, model_name: str):
        self.__model_name = model_name + "_PyTorch"

        model = self.load_results()
        print(["... ANN is loaded: ", self.__model_name])
        self.__model_trained = True
        return model


    def load_data_direct(self, train_in: np.ndarray, train_out: np.ndarray, valid_in: np.ndarray, valid_out: np.ndarray, do_norm: bool):
        self.__data_input = None
        self.__data_output = None

        # --- Normalization of the data
        # TODO: Adding normalization of the data (1st: float, [-1, +1] - 2nd: int, fullscale adc in FPGA)
        if do_norm:
            # Xin = self.__norm_data(train_in.astype("double"))
            # Yin = self.__norm_data(valid_in.astype("double"))
            # Xout = self.__norm_data(train_out.astype("double"))
            # Yout = self.__norm_data(valid_out.astype("double"))

            val_max = np.zeros(shape=(4, 1))
            val_max[0] = np.abs(train_in).max()
            val_max[1] = np.abs(train_out).max()
            val_max[2] = np.abs(valid_in).max()
            val_max[3] = np.abs(valid_out).max()

            val_norm = val_max.max()
            print("... Normierungsfaktor von:", val_norm)
            Xin = train_in.astype("double")/val_norm
            Yin = valid_in.astype("double")/val_norm
            Xout = train_out.astype("double")/val_norm
            Yout = valid_out.astype("double")/val_norm
        else:
            Xin = train_in.astype("double")
            Yin = valid_in.astype("double")
            Xout = train_out.astype("double")
            Yout = valid_out.astype("double")

        # --- Converting data format from NumPy to Torch.Tensor
        self.__train_input = torch.from_numpy(Xin)
        self.__valid_input = torch.from_numpy(Yin)
        self.__train_output = torch.from_numpy(Xout)
        self.__valid_output = torch.from_numpy(Yout)

    def load_data_split(self, input: np.ndarray, output: np.ndarray, do_norm: bool):
        self.__data_input = input.astype("double")
        self.__data_output = output.astype("double")

        # --- Normalization of the data
        # (1st: float, [-1, +1] - 2nd: int, fullscale adc in FPGA)
        # TODO: Select the normalization mode
        if do_norm:
            self.__data_input = self.__norm_data(self.__data_input)
            self.__data_output = self.__norm_data(self.__data_output)

        # --- Splitting the datasets into training and validation
        (Xin, Yin, Xout, Yout) = train_test_split(
            self.__data_input, self.__data_output,
            train_size=self.train_size,
            test_size=self.valid_size,
            shuffle=self.do_shuffle_data
        )

        # --- Converting data format from NumPy to Torch.Tensor
        self.__train_input = torch.from_numpy(Xin)
        self.__valid_input = torch.from_numpy(Yin)
        self.__train_output = torch.from_numpy(Xout)
        self.__valid_output = torch.from_numpy(Yout)

    def get_train_data(self):
        train_in = self.__train_input
        train_out = self.__train_output
        valid_in = self.__valid_input
        valid_out = self.__valid_output

        return (train_in, train_out, valid_in, valid_out)


    def prepare_quantization(self):
        # TODO: Add ElasticAI.creator for quantization
        return 42


    def print_model(self, mode: bool = False):
        print("... printing structure and parameters of the neural network")
        if(mode == False):
            summary(self.model, (self.model_input_size,))
        else:
            print(self.model)
            for name, param in self.model.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def do_training(self, batch_size: float, epochs: int, learning_rate: float):
        # TODO: Implement other optimizer over definition
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        losses_train = []
        losses_valid = []
        print("... start training")

        # ---- Training
        train_in = self.__train_input.to(self.device)
        train_out = self.__train_output.to(self.device)
        size = len(train_in)

        print("-------------------------------")
        for epoch in range(epochs):
            lossTrain = 0
            #for (batch, x_in) in enumerate(train_in):
            self.model.train()
            (_, y_pred) = self.model(train_in)
            #y_soll = train_out[batch]
            y_soll = train_out
            lossTrain = loss_function(y_pred, y_soll)

            # Backpropagation
            optimizer.zero_grad()
            lossTrain.backward()
            optimizer.step()

            losses_train.append(lossTrain.item())

            # ---- Validation
            valid_in = self.__valid_input.to(self.device)
            valid_out = self.__valid_output.to(self.device)
            size = len(valid_in)

            lossValid = 0
            self.model.eval()
            with torch.no_grad():
                for (batch, x_in) in enumerate(valid_in):
                    (_, y_pred) = self.model(x_in)
                    lossValid += loss_function(y_pred, valid_out[batch]).item()

            lossValid /= size
            losses_valid.append(lossValid)
            print(f"Epoch: {epoch+1} - Loss (Train): {lossTrain:>7f} - Loss (Valid): {lossValid:>7f} [{size:>5d}]")

        self.__model_trained = True
        print("... ANN model is trained")
        pass

    def do_prediction(self, x_input):
        if(self.__model_trained):
            #x0 = torch.from_numpy(x_input)
            x0 = x_input
            (Feat, Yout) = self.model(x0)
        else:
            Feat = []
            Yout = []
            print("... model not loaded - Please check before running prediction")

        return (Feat.detach(), Yout.detach())

    def save_results(self):
        self.__save_model(self.__path2models, self.__model_name)

    def load_results(self) -> None:
        path2file = os.path.join(self.__path2models, self.__model_name)
        if os.path.exists(path=path2file):
            self.__load_model(self.__path2models, self.__model_name)
            self.__model_loaded = True
            self.__model_trained = True
        else:
            self.__model_loaded = False
            self.__model_trained = False
            print(" --- MODEL NOT AVAILABLE - Check the naming or path!")
            sys.exit()

    def __load_model(self, path2model: str, name: str) -> None:
        load_model = os.path.join(path2model, name)
        self.model = torch.load(load_model)

    def __save_model(self, path2model: str, name: str) -> None:
        save_file = os.path.join(path2model, name)
        torch.save(self.model, save_file)

    def __setup(self):
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
