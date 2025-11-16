import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch import nn as nn_torch
from elasticai.creator.nn import fixed_point as nn_creator
from denspp.offline import get_path_to_project
from denspp.offline.plot_helper import save_figure
from denspp.offline.analog.common_func import CommonDigitalFunctions


if __name__ == "__main__":
    # --- Settings
    val_max = 4.
    total_bits = 5

    # --- Definitions
    stimulus = np.linspace(start=-val_max, stop=+val_max, num=51, endpoint=True)
    range = (float(stimulus.min()), float(stimulus.max()))
    frac_bits = total_bits-int(np.log2(val_max))-1
    func = CommonDigitalFunctions()
    func.define_limits(True, total_bits, frac_bits)
    #stimulus = func.quantize_fxp(stimulus)

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 4))
    for idx, ax in enumerate(axs):
        match idx:
            case 0:
                act0 = nn_torch.ReLU()
                out0 = act0(Tensor(stimulus))
                act1 = nn_creator.ReLU(total_bits)
                out1 = act1(Tensor(stimulus))
                act2 = nn_creator.ReLU(total_bits, True)
                out2 = act2(Tensor(stimulus))
                ax.set_title('ReLU')
                ax.set_yticks([0., 1., 2., 3., 4.])
                ax.set_ylim([-0.02, val_max+0.02])
            case 1:
                act0 = nn_torch.Tanh()
                out0 = act0(Tensor(stimulus))
                act1 = nn_creator.Tanh(total_bits, frac_bits, 21)
                out1 = act1(Tensor(stimulus))
                act2 = nn_creator.HardTanh(total_bits, frac_bits)
                out2 = act2(Tensor(stimulus))
                ax.set_title('Tanh')
                ax.set_yticks([-1., -.5, 0., .5, 1.])
                ax.set_ylim([-1.02, +1.02])
            case _:
                act0 = nn_torch.Sigmoid()
                out0 = act0(Tensor(stimulus))
                act1 = nn_creator.Sigmoid(total_bits, frac_bits, 21)
                out1 = act1(Tensor(stimulus))
                act2 = nn_creator.HardSigmoid(total_bits, frac_bits)
                out2 = act2(Tensor(stimulus))
                ax.set_title('Sigmoid')
                ax.set_yticks([0., .25, .5, .75, 1.])
                ax.set_ylim([-0.02, +1.02])

        ax.plot(stimulus, out0, color='k', label='Full')
        ax.step(stimulus, out1, color='r', label='Quant (FxP)', where="mid")
        ax.step(stimulus, out2, color='g', label='Hard (FxP)', where="mid")
        ax.grid()

    axs[1].legend(loc='upper left')
    axs[0].set_xlim(range)
    axs[0].set_xticks([range[0], range[0]/2, 0., range[1]/2, range[1]])
    axs[0].set_ylabel('Output')
    axs[1].set_xlabel('Input')

    plt.tight_layout()
    save_figure(fig, path=f"{get_path_to_project()}/runs", name="quant_act")
    plt.show()
