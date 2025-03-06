from torch import nn, Tensor, argmax, flatten
from elasticai.creator.nn import Sequential as SequantialCreator
from elasticai.creator.nn.fixed_point import Linear, BatchNormedLinear, Tanh, ReLU


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
    def __init__(self, input_size: int=32, output_size: int=3,
                 total_bits: int=12, frac_bits: int = 8, num_steps_activation: int = 32):
        super().__init__()
        self.bit_config = [total_bits, frac_bits]
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = SequantialCreator(
            BatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
            Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps_activation),
            BatchNormedLinear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias, total_bits=total_bits, frac_bits=frac_bits, bn_affine=do_train_batch),
        )
        # --- Decoder Path
        self.decoder = SequantialCreator(
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
        self.encoder = SequantialCreator(
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
        self.decoder = SequantialCreator(
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


class CompareDNN_Classifier_v1_Torch(nn.Module):
    def __init__(self, input_size: int=32, output_size: int=6):
        """DL model for classifying neural spike activity (MLP)"""
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 12, output_size]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}",
                                  nn.Linear(in_features=config_network[idx - 1], out_features=layer_size,
                                            bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}",
                                  nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network) - 1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                # self.model.add_module(f"soft", nn.Softmax(dim=1))
                pass

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


class CompareDNN_Classifier_v1_Creator(nn.Module):
    def __init__(self, input_size: int=32, output_size:int=6,
                 total_bits: int = 12, frac_bits: int = 8):
        """DL model for classifying neural spike activity (MLP)"""
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 12, output_size]

        # --- Model Deployment
        self.model = SequantialCreator()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}",
                                  BatchNormedLinear(
                                      in_features=config_network[idx - 1],
                                      out_features=layer_size,
                                      bias=do_train_bias,
                                      total_bits=total_bits, frac_bits=frac_bits
                                  ))
            if not idx == len(config_network) - 1:
                self.model.add_module(f"act_{idx:02d}", ReLU(total_bits=total_bits))
            else:
                pass

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)