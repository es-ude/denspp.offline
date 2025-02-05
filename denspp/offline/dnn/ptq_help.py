import numpy as np
from torch import nn, Tensor, abs, sum
from copy import deepcopy
from elasticai.creator.nn.fixed_point.quantization import quantize

from denspp.offline.dnn.pytorch_config_model import DefaultSettingsTrainMSE
from denspp.offline.dnn.pytorch_config_data import DefaultSettingsDataset


def quantize_model_fxp(model: nn.Sequential, total_bits: int, frac_bits: int) -> nn.Module:
    """Function for quantizing the model parameters
    :param model:       Torch model / Sequential to be quantized
    :param total_bits:  Total number of bits
    :param frac_bits:   Fraction of bits to quantize
    :return:            Quantized model
    """
    model_quant = deepcopy(model)
    for name, sequential in model_quant.named_children():
        seq_quant = nn.Sequential()
        for layer in sequential.children():
            if hasattr(layer, 'bias'):
                layer.bias.data = quantize(layer.bias.data, total_bits, frac_bits)
            if hasattr(layer, 'weight'):
                layer.weight.data = quantize(layer.weight.data, total_bits, frac_bits)
            seq_quant.append(layer)
        model_quant.add_module(name, seq_quant)
    return model_quant


def quantize_data_fxp(data: Tensor | np.ndarray, total_bits: int, frac_bits: int) -> Tensor:
    """Function for quantizing the data
    :param data:        Tensor data to be quantized
    :param total_bits:  Total number of bits
    :param frac_bits:   Fraction of bits to quantize
    :return:            Quantized Tensor data
    """
    data_used = data if isinstance(data, Tensor) else Tensor(data)
    return quantize(data_used, total_bits=total_bits, frac_bits=frac_bits)


# TODO: Do test?
if __name__ == "__main__":
    settings_test = DefaultSettingsTrainMSE
    settings_test.model_name = 'CompareDNN_Autoencoder_v1_Torch'
    model_test = settings_test.get_model()
    model_test.eval()

    settings_data = DefaultSettingsDataset
    settings_data.data_file_name = 'quiroga'
    settings_data.normalization_do = True
    dataset = settings_data.load_dataset()

    model_qunt = quantize_model_fxp(model_test,12, 10)
    model_qunt.eval()

    data_input = Tensor(dataset['data'])
    dout_test = model_test(data_input)
    dout_qunt = model_qunt(data_input)

    dmae = sum(abs(dout_test - dout_qunt)).detach().numpy() / len(data_input)
    mae_loss = sum(abs(data_input - dout_qunt[1])).detach().numpy() / len(data_input)
    print(mae_loss, dmae)
