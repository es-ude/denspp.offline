import numpy as np
from torch import nn, Tensor, no_grad
from copy import deepcopy
from elasticai.creator.arithmetic import FxpParams, FxpArithmetic
from elasticai.creator.nn.fixed_point import MathOperations


def quantize_model_fxp(model: nn.Sequential, total_bits: int, frac_bits: int) -> nn.Module:
    """Function for quantizing all model parameters
    :param model:       Torch model / Sequential to be quantized
    :param total_bits:  Total number of bits
    :param frac_bits:   Fraction of bits to quantize
    :return:            Quantized model
    """
    fxpmath = MathOperations(FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)))
    model_quant = deepcopy(model)
    with no_grad():
        for name, param in model_quant.named_parameters():
            param.copy_(fxpmath.quantize(param))
    return model_quant


def quantize_data_fxp(data: Tensor | np.ndarray, total_bits: int, frac_bits: int) -> Tensor:
    """Function for quantizing the data
    :param data:        Tensor data to be quantized
    :param total_bits:  Total number of bits
    :param frac_bits:   Fraction of bits to quantize
    :return:            Quantized Tensor data
    """
    fxpmath = MathOperations(FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)))
    data_used = data if isinstance(data, Tensor) else Tensor(data)
    return fxpmath.quantize(data_used)
