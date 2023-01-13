import tensorflow.keras as nntf
from tensorflow import Tensor

class nn_autoencoder (nntf.Model):
    def __init__(self, io_size: int):
        super().__init__()
        self.nHiddenLayers = [io_size, 24, 16, 8, 16, 24, io_size]
        self.init_w = nntf.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.init_b = nntf.initializers.Constant(value=0.0)

        self.model0 = nntf.models.Sequential()
        # --- Encoder Path
        for idx in self.nHiddenLayers:
            self.model0.add(nntf.layers.Dense(input_shape=(idx,), units=idx))
            if idx != self.nHiddenLayers:
                self.model0.add(nntf.layers.Activation("tanh"))