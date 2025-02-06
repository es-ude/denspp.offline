import numpy as np
from torch import nn, Tensor
from copy import deepcopy
from elasticai.creator.nn.fixed_point.quantization import quantize


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
