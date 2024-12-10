from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from torch import nn, Tensor
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, Tanh, BatchNormedLinear
from package.dnn.model_library import ModelRegistry


models_bib = ModelRegistry()


@models_bib.register
class synthetic_dnn_ae_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.model_shape = (1, input_size)
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[3], affine=do_train_batch),
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

    def create_encoder_design(self, name):
        return self.encoder.create_design(name)


@models_bib.register
class synthetic_dnn_ae_v2(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.model_shape = (1, input_size)
        iohiddenlayer = [input_size, 20, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


@models_bib.register
class synthetic_dnn_ae_v1_quantized(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.model_shape = (1, input_size)
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True
        bitwidth_total = 16
        bitwidth_frac = 14

        # --- Encoder Path
        self.encoder = Sequential(
            BatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
            Tanh(total_bits=16, frac_bits=2, num_steps=32),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
            Tanh(total_bits=16, frac_bits=2, num_steps=32),
            BatchNormedLinear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
        )
        # --- Decoder Path
        self.decoder = Sequential(
            Tanh(total_bits=bitwidth_total, frac_bits=bitwidth_frac, num_steps=32),
            BatchNormedLinear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
            Tanh(total_bits=16, frac_bits=2, num_steps=32),
            BatchNormedLinear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
            Tanh(total_bits=16, frac_bits=2, num_steps=32),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias,
                              total_bits=bitwidth_total, frac_bits=bitwidth_frac, bn_affine=do_train_batch),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

    def create_encoder_design(self, name):
        return self.encoder.create_design(name)


@models_bib.register
class synthethic_dnn_ae_v2(nn.Module):
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.model_shape = (1, input_size)
        bits_total = 12
        bits_frac = 9
        iohiddenlayer = [input_size, 20, output_size]
        do_train_bias = True

        # --- Encoder Path
        self.encoder = Sequential(
            BatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1],
                              total_bits=bits_total, frac_bits=bits_frac,
                              bias=do_train_bias),
            Tanh(total_bits=bits_total, frac_bits=bits_frac),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2],
                              total_bits=bits_total, frac_bits=bits_frac,
                              bias=do_train_bias)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[2]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2],
                      out_features=iohiddenlayer[1]),
            nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1],
                      out_features=iohiddenlayer[0])
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)
