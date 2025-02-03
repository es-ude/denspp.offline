from torch import nn, Tensor
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, BatchNormedLinear, Tanh


class CompareDNN_Autoencoder_v1_Torch(nn.Module):
    def __init__(self, input_size: int = 32, output_size: int = 3):
        super().__init__()
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(iohiddenlayer[1], affine=True),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(iohiddenlayer[2], affine=True),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(iohiddenlayer[2], affine=True),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(iohiddenlayer[1], affine=True),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class CompareDNN_Autoencoder_v1_Creator(nn.Module):
    def __init__(self, input_size=32, output_size=3, total_bits: int=12, frac_bits: int = 8, num_steps_activation: int = 32):
        super().__init__()
        self.bit_config = [total_bits, frac_bits]
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = Sequential(
            BatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
        )
        # --- Decoder Path
        self.decoder = Sequential(
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

    def forward_first_layer(self, x: Tensor) -> Tensor:
        return self.encoder[0](x)

    def create_encoder_design(self, name):
        return self.encoder.create_design(name)


class CompareDNN_Autoencoder_woBN_v1_Torch(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size: int=32, output_size: int=3):
        super().__init__()
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class CompareDNN_Autoencoder_woBN_v1_Creator(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size: int = 32, output_size: int = 3, total_bits: int = 12, frac_bits: int = 8, num_steps_activation: int = 32):
        super().__init__()
        self.bit_config = [total_bits, frac_bits]
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True


        # --- Encoder Path
        self.encoder = Sequential(
            Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
        )
        # --- Decoder Path
        self.decoder = Sequential(
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias,
                   total_bits=total_bits, frac_bits=frac_bits),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

    def forward_first_layer(self, x: Tensor) -> Tensor:
        return self.encoder[0](x)

    def create_design(self, name):
        encoder = self.encoder.create_design(f"{name}_encoder")
        decoder = self.decoder.create_design(f"{name}_decoder")
        return encoder, decoder
