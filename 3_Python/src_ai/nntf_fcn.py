import tensorflow.keras as nntf
from tensorflow import Tensor


class nn_autoencoder (nntf.Model):
    def __init__(self, io_size: int):
        super().__init__()
        self.nHiddenLayers_encode = [io_size, 32, 20, 8]
        self.nHiddenLayers_decode = [4, 8, 20, 32, io_size]
        self.init_w = nntf.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        # self.init_b = nntf.initializers.Constant(value=0.0)

        self.model0 = nntf.models.Sequential()
        # --- Encoder Path
        for idx in self.nHiddenLayers_encode:
            self.model0.add(nntf.layers.Conv1D(idx, 16, 2, padding='SAME',  input_shape=(1,40)))
            self.model0.add(nntf.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'))
            self.model0.add(nntf.layers.BatchNormalization())
            self.model0.add(nntf.layers.ELU())



        for idx in self.nHiddenLayers_decode:
            self.model0.add(nntf.layers.Conv1DTranspose(idx, 16, 2, padding='SAME'))#, kernel_regularizer = regularizer)
            self.model0.add(nntf.layers.BatchNormalization())
            self.model0.add(nntf.layers.ELU())



    def forward(self, features: Tensor) -> Tensor:
        y_pred = self.model0(features)

        return y_pred