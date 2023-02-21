import tensorflow.keras as nntf
from tensorflow import Tensor

class nn_dnn_autoencoder (nntf.Model):
    def __init__(self, io_size: int):
        super().__init__()
        self.model_name = "dnn_dae_v2"
        self.nHiddenLayers = [28, 12, 28]
        # self.init_w = nntf.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        # self.init_b = nntf.initializers.Constant(value=0.0)

        self.model0 = nntf.models.Sequential()
        self.model0.add(nntf.layers.Input(shape=(io_size)))
        for idx in self.nHiddenLayers:
            self.model0.add(nntf.layers.Dense(idx))
            self.model0.add(nntf.layers.Activation("tanh"))

        self.model0.add(nntf.layers.Dense(io_size))


class nn_cnn_autoencoder (nntf.Model):
    def __init__(self, io_size: int):
        super().__init__()
        self.model_name = "cnn_dae_v2"
        # self.init_w = nntf.initializers.RandomUniform(minval=-0.05, maxval=0.05)

        self.model0 = nntf.models.Sequential()
        inputs = nntf.layers.Input(shape=(40, 1))
        # Encoder setup
        encoded = nntf.layers.Conv1D(32, 3, padding='same')(inputs)
        encoded = nntf.layers.BatchNormalization()(encoded)
        encoded = nntf.layers.Activation('relu')(encoded)

        encoded = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        encoded = nntf.layers.Conv1D(16, 3, padding='same')(encoded)
        encoded = nntf.layers.Activation('relu')(encoded)

        encoded = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        encoded = nntf.layers.Conv1D(8, 3, padding='same')(encoded)
        encoded = nntf.layers.Activation('relu')(encoded)

        encoded = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        # Decoder setup
        decoded = nntf.layers.Conv1D(8, 3, padding='same')(encoded)
        decoded = nntf.layers.Activation('relu')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)

        decoded = nntf.layers.Conv1D(16, 3, padding='same')(decoded)
        decoded = nntf.layers.Activation('relu')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)

        decoded = nntf.layers.Conv1D(32, 3, padding='same')(decoded)
        decoded = nntf.layers.Activation('relu')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)

        decoded = nntf.layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)

        self.model0 = nntf.models.Model(inputs, decoded)
