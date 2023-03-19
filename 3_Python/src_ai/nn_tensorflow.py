import os, time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from IPython.display import clear_output
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src_ai.nntf_architecture import dnn_autoencoder as nn_topology

class NeuralNetwork (nn_topology):
    def __init__(self):
        nn_topology.__init__(self)
        # --- Properties
        self.os_type = os.name
        self.device = self.__setup()
        self.__path2models = "models"
        self.__path2logs = "logs"
        self.__path2fig = "figures"
        # --- Properties (Model)
        self.model_input_size = 0
        self.__model_name = None
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
        pass
        #self.model = model
        #self.model_input_size = input_size

    def initTrain(self, train_size: float, valid_size: float, shuffle: bool, name_addon: str):
        self.__model_name = self.set_name + name_addon + "_TensorFlow"

        self.train_size = train_size
        self.valid_size = valid_size
        self.do_shuffle_data = shuffle

    def initPredict(self, model_name: str):
        self.__model_name = model_name + "_Tensorflor"

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
            # Xin = self.__norm_data(train_in.astype("float"))
            # Yin = self.__norm_data(valid_in.astype("float"))
            # Xout = self.__norm_data(train_out.astype("float"))
            # Yout = self.__norm_data(valid_out.astype("float"))

            val_max = np.zeros(shape=(4, 1))
            val_max[0] = np.abs(train_in).max()
            val_max[1] = np.abs(train_out).max()
            val_max[2] = np.abs(valid_in).max()
            val_max[3] = np.abs(valid_out).max()

            val_norm = val_max.max()
            print("... Normierungsfaktor von:", val_norm)
            Xin = train_in.astype("float")/val_norm
            Yin = valid_in.astype("float")/val_norm
            Xout = train_out.astype("float")/val_norm
            Yout = valid_out.astype("float")/val_norm
        else:
            Xin = train_in.astype("float")
            Yin = valid_in.astype("float")
            Xout = train_out.astype("float")
            Yout = valid_out.astype("float")

        # --- Converting data format from NumPy to Torch.Tensor
        self.__train_input = tf.convert_to_tensor(Xin)
        self.__valid_input = tf.convert_to_tensor(Yin)
        self.__train_output = tf.convert_to_tensor(Xout)
        self.__valid_output = tf.convert_to_tensor(Yout)

    def load_data_split(self, input: np.ndarray, output: np.ndarray, do_norm: bool):
        self.__data_input = input.astype("float")
        self.__data_output = output.astype("float")

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
        self.__train_input = tf.convert_to_tensor(Xin)
        self.__valid_input = tf.convert_to_tensor(Yin)
        self.__train_output = tf.convert_to_tensor(Xout)
        self.__valid_output = tf.convert_to_tensor(Yout)

    def get_train_data(self):
        train_in = self.__train_input
        train_out = self.__train_output
        valid_in = self.__valid_input
        valid_out = self.__valid_output

        return (train_in, train_out, valid_in, valid_out)


    def print_model(self):
        print("... printing structure and parameters of the neural network")
        self.model.summary()

    def do_training(self):
        # conf_mat = ConfusionMatrix(
        #     self.model,
        #     self.__valid_input,
        #     self.__valid_output,
        #     classes_list=classes_list,
        #     log_dir=self.__path2logs
        # )
        callbacks_list = [PlotLearning(self.set_epochs, self.__path2fig, self.set_name)]

        # Overview of optimizer:    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        # Overview of Losses:       https://www.tensorflow.org/api_docs/python/tf/keras/losses
        # Overview of Metrics:      https://www.tensorflow.org/api_docs/python/tf/keras/metrics
        self.model.compile(
            optimizer=self.set_optimizer,
            loss=self.set_loss,
            metrics=self.set_metric
        )

        # --- Training phase
        start_time = time.time()
        print("... start training")
        # verbose = 0: None, 1: progress bar, 2: one line per epoch
        self.model.fit(
            x=self.__train_input,
            y=self.__train_output,
            epochs=self.set_epochs,
            batch_size=self.set_batchsize,
            verbose=1,
            validation_data=(self.__valid_input, self.__valid_output),
            callbacks=[callbacks_list],
            shuffle=self.do_shuffle_data,
            use_multiprocessing=False
        )

        end_time = time.time()
        run_time = (end_time - start_time)
        print(f'... training done after runtime of {run_time} s')

        # --- Validation phase
        self.score = self.model.evaluate(
            x=self.__valid_input,
            y=self.__valid_output
        )
        self.__model_trained = True
        print(f"... ANN model is trained with score: {self.score}")

    def do_prediction(self, x_input):
        if(self.__model_trained):
            Feat = self.encoder(x_input)
            Yout = self.decoder(Feat)
        else:
            Feat = []
            Yout = []
            print("... model not loaded - Please check before running prediction")

        return Feat, Yout


    def save_results(self, model_save: bool = True):
        path2file = os.path.join(self.__path2models, self.__model_name)
        if model_save:
            self.model.save(filepath=path2file)
        else:
            self.model.save_weights(filepath=path2file)

    def load_results(self):
        path2file = os.path.join(self.__path2models, self.__model_name)
        if os.path.exists(path=path2file):
            model = tf.keras.models.load_model(filepath=path2file)
        else:
            model = None
            print(" --- MODEL NOT AVAILABLE - Check the naming or path!")
            sys.exit()
        return model

    def __setup(self):
        # TODO: Differenzierung für Betriebssysteme einfügen
        if self.os_type == "nt":
            ostype = "Windows"
        elif self.os_type == "mac":
            ostype = "MAC"
        elif self.os_type == "posix":
            ostype = "Linux"

        # TODO: Prüfen, ob CUDA verfügbar ist
        device0 = "CPU"
        # tf.config.LogicalDevice(device_type="CPU")
        # tf.config.LogicalDeviceConfiguration(memory_limit=None)

        print("... using TensorFlow with", device0, "device on", ostype)
        return device0

    def __plot_metrics(self):
        # This is done with separate callback class "PlotLearning"
        pass

    # TODO: Bessere Methode implementieren --> Methode bisher ist behindert
    def __norm_data(self, frames_in: np.ndarray) -> np.ndarray:
        max_val = np.argmax(frames_in)
        min_val = np.argmin(frames_in)
        frames_out = frames_in / max_val

        # frames_out = np.zeros(shape=frames_in.shape)
        # for idx in range(frames_in.shape[0]):
        #    frame = frames_in[idx, :]
        #    frames_out[idx, :] = frame/np.max(np.abs(frame))
        return frames_out


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def __init__(self, noEpoch: int, path2figures: str, model_name: str):
        self.__no_epoch = noEpoch
        self.__fig_size = (12, 5)
        self.__path2figs = path2figures
        self.__name = model_name

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = [x for x in logs if 'val' not in x]


        if epoch == (self.__no_epoch-1):
            f, axs = plt.subplots(1, len(metrics), sharex=True, figsize=self.__fig_size)
            clear_output(wait=True)

            for i, metric in enumerate(metrics):
                axs[i].plot(
                    range(1, epoch + 2),
                    self.metrics[metric],
                    label=metric,
                    marker='.'
                )
                if logs['val_' + metric]:
                    axs[i].plot(
                        range(1, epoch + 2),
                        self.metrics['val_' + metric],
                        label='val_' + metric,
                        marker='.'
                    )

                axs[i].legend()
                axs[i].set_xlabel("Epochs")
                axs[i].set_ylabel("Metric")
                axs[i].grid()

            plt.tight_layout()
            plt.xlim([0, self.__no_epoch+1])
            plt.xticks(np.linspace(1, self.__no_epoch, 11, dtype='int'))
            plt.show(block=False)

            # --- Saving Figure
            dateToday = date.today().strftime("%Y%m%d")
            filename = dateToday + "_" + self.__name + "_TrainTF"
            figname = os.path.join(self.__path2figs, filename)
            plt.savefig(figname+'.jpg', transparent=True, format='jpg')
            plt.savefig(figname+'.eps', transparent=True, format='eps')