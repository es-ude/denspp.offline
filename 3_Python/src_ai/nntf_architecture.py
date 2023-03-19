import tensorflow.keras as nntf

# Infos zum Einstellen des TF-Compilers
# Overview of optimizer:    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# Overview of Losses:       https://www.tensorflow.org/api_docs/python/tf/keras/losses
# Overview of Metrics:      https://www.tensorflow.org/api_docs/python/tf/keras/metrics

# Anmerkungen zum Optimizer für Denoising Autoencoder mit Feature-Bestimmung
# - rmsprop führt zu guten Ergebnissen, aber Feature Extraction ist nicht möglich

class dnn_autoencoder(nntf.Model):
    def __init__(self):
        super().__init__()
        self.set_name = "dnn_dae_v2"
        self.__io_size = 40
        self.set_batchsize = 2
        self.set_epochs = 300
        self.set_learningrate = 1e-3
        self.set_optimizer = "sgd"
        # rmsprop führt zu guten Ergebnissen, aber Feature Extraction ist nicht möglich
        # self.set_optimizer = "rmsprop"
        self.set_loss = "mae"
        self.set_metric = ["mse", "cosine_similarity"]

        self.nHiddenLayers = [20, 8, 20]

        # Input
        input = nntf.layers.Input(shape=(self.__io_size))
        # Encoder
        encoded = nntf.layers.Dense(self.nHiddenLayers[0])(input)
        encoded = nntf.layers.Activation("tanh")(encoded)
        encoded = nntf.layers.Dense(self.nHiddenLayers[1])(encoded)
        encoded0 = nntf.layers.Activation("tanh")(encoded)
        # Decoder
        decoded = nntf.layers.Dense(self.nHiddenLayers[2])(encoded0)
        decoded = nntf.layers.Activation("tanh")(decoded)
        decoded = nntf.layers.Dense(self.__io_size)(decoded)
        decoded0 = nntf.layers.Activation("tanh")(decoded)

        self.encoder = nntf.Model(input, encoded0)
        self.decoder = nntf.Model(encoded0, decoded0)
        self.model = nntf.Model(input, decoded0)

class cnn_autoencoder(nntf.Model):
    def __init__(self):
        super().__init__()
        self.set_name = "cnn_dae_v2"
        self.__optimizer_param = None
        self.__io_size = 40
        self.set_batchsize = 16
        self.set_epochs = 300
        self.set_learningrate = 1e-3
        self.set_optimizer = "rmsprop"
        self.set_loss = "mae"
        self.set_metric = ["mse", "cosine_similarity"]

        input = nntf.layers.Input(shape=(self.__io_size, 1))
        # Encoder setup
        encoded = nntf.layers.Conv1D(32, 3, padding='same')(input)
        encoded = nntf.layers.BatchNormalization()(encoded)
        encoded = nntf.layers.Activation('tanh')(encoded)
        encoded = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        encoded = nntf.layers.Conv1D(16, 3, padding='same')(encoded)
        encoded = nntf.layers.Activation('tanh')(encoded)
        encoded = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        encoded = nntf.layers.Conv1D(8, 3, padding='same')(encoded)
        encoded = nntf.layers.Activation('tanh')(encoded)
        encoded0 = nntf.layers.MaxPooling1D(2, padding='same')(encoded)
        # Decoder setup
        decoded = nntf.layers.Conv1D(8, 3, padding='same')(encoded0)
        decoded = nntf.layers.Activation('tanh')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)
        decoded = nntf.layers.Conv1D(16, 3, padding='same')(decoded)
        decoded = nntf.layers.Activation('tanh')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)
        decoded = nntf.layers.Conv1D(32, 3, padding='same')(decoded)
        decoded = nntf.layers.Activation('tanh')(decoded)
        decoded = nntf.layers.UpSampling1D(2)(decoded)
        decoded0 = nntf.layers.Conv1D(1, 3, padding='same')(decoded)

        self.encoder = nntf.Model(input, encoded0)
        self.decoder = nntf.Model(encoded0, decoded0)
        self.model = nntf.models.Model(input, decoded0)
