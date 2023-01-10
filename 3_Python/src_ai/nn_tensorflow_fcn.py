import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from datetime import date


from sklearn.model_selection import train_test_split
# from src_ai.nntf_architecture import nn_autoencoder
from src_ai.nntf_fcn import nn_autoencoder

class NeuralNetwork (nn_autoencoder):
    def __init__(self, train_size: float, valid_size: float, shuffle: bool, input_size: int, model_name: str):
        nn_autoencoder.__init__(self, input_size)
        # --- Properties
        self.os_type = os.name
        self.device = self.__setup()
        self.__model_name = model_name
        self.__path2models = "models"
        self.__path2logs = "logs"
        self.__path2fig = "figures_fcn"
        # --- Definition of the data
        self.__data_input = None
        self.__data_output = None
        self.train_size = train_size
        self.valid_size = valid_size
        self.do_shuffle_data = shuffle
        # --- Splitting into training and validation datasets
        self.__train_input = None
        self.__train_output = None
        self.__valid_input = None
        self.__valid_output = None
        # --- Model instanziation
        self.__model_loaded = False
        self.model = self.model0
        self.__model_trained = False
        self.model.build((1,40))

    def load_data(self, input: np.ndarray, output: np.ndarray, do_norm: bool = False):
        self.__data_input = input
        self.__data_output = output

        # --- Normalization of the data
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
        # self.__train_input = Xin.astype("float").reshape((len(Xin), 1,40))
        # self.__valid_input = Yin.astype("float").reshape((len(Yin), 1,40))
        # self.__train_output = Xout.astype("float").reshape((len(Xout), 1,40))
        # self.__valid_output = Yout.astype("float").reshape((len(Yout), 1,40))
        self.__train_input = tf.convert_to_tensor(Xin.astype("float").reshape((len(Xin), 1,40)))
        self.__valid_input = tf.convert_to_tensor(Yin.astype("float").reshape((len(Yin), 1,40)))
        self.__train_output = tf.convert_to_tensor(Xout.astype("float").reshape((len(Xout), 1,40)))
        self.__valid_output = tf.convert_to_tensor(Yout.astype("float").reshape((len(Yout), 1,40)))
        

    def print_model(self):
        print("... printing structure and parameters of the neural network")
        self.model.summary()

    def train_model(self, batch_size: float, epochs: int, learning_rate: float):
        # conf_mat = ConfusionMatrix(
        #     self.model,
        #     self.__valid_input,
        #     self.__valid_output,
        #     classes_list=classes_list,
        #     log_dir=self.__path2logs
        # )
        callbacks_list = [PlotLearning(epochs, self.__path2fig)]

        # Overview of Losses:       https://www.tensorflow.org/api_docs/python/tf/keras/losses
        # Overview of Metrics:      https://www.tensorflow.org/api_docs/python/tf/keras/metrics
        # Overview of optimizer:    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        self.model.compile(
            optimizer="Adamax",
            loss=["mae"],
            metrics=["accuracy", "mean_absolute_error"]
        )
        # verbose = 0: None, 1: progress bar, 2: one line per epoch
        self.model.fit(
            x=self.__train_input,
            y=self.__train_output,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(self.__valid_input, self.__valid_output),
            callbacks=[callbacks_list],
            shuffle=self.do_shuffle_data,
            use_multiprocessing=False
        )
        self.score = self.model.evaluate(
            x=self.__valid_input,
            y=self.__valid_output
        )
        self.__model_trained = True
        print(f"... ANN model is trained with score: {self.score}")

    def save_results(self, model_save: bool):
        path2file = os.path.join(self.__path2models, self.__model_name)
        if model_save:
            self.model.save(filepath=path2file)
        else:
            self.model.save_weights(filepath=path2file)

    def get_train_data(self):
        valid_in = self.__valid_input
        valid_out = self.__valid_output

        return (valid_in, valid_out)

    def predict_model(self, x_input: np.ndarray):
        if(self.__model_trained):
            Yout = self.model.predict(
                x=x_input,
                use_multiprocessing=True
            )
        else:
            Yout = []
            print("... model not loaded - Please check before running prediction")

        return Yout

    def load_results(self):
        path2file = os.path.join(self.__path2models, self.__model_name)

        self.model = tf.keras.models.load_model(filepath=path2file)

    def __setup(self):
        # TODO: Differenzierung für Betriebssysteme einfügen
        if self.os_type == "nt":
            ostype = "Windows"
        elif self.os_type == "mac":
            ostype = "MAC"
        elif self.os_type == "posix":
            ostype = "Linux"

        # TODO: Prüfen ob CUDA verfügbar ist
        device0 = "CPU"
        #tf.config.LogicalDevice(device_type="CPU")
        tf.config.LogicalDeviceConfiguration(memory_limit=None)

        print("... using TensorFlow with", device0, "device on", ostype)
        return device0

    def __plot_metrics(self):
        pass

    def __norm_data(self, frames_in: np.ndarray) -> np.ndarray:
        frames_out = frames_in
        for idx in range(frames_in.shape[0]):
            print(idx, frames_in.shape[0])
            frame = frames_in[idx,:]
            frames_out[idx,:] = frame/np.max(np.abs(frame))

        return frames_out
class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def __init__(self, noEpoch: int, path2figures: str):
        self.__no_epoch = noEpoch
        self.__fig_size = (12,5)
        self.__path2figs = path2figures

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
            figname = os.path.join(self.__path2figs, dateToday + "_TrainingResults_PyTorch")
            plt.savefig(figname+'.jpg', transparent=True, format='jpg')
            plt.savefig(figname+'.eps', transparent=True, format='eps')
