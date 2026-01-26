import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, absolute, sign, heaviside, zeros_like
from torch import nn as nn_torch
from elasticai.creator.nn import fixed_point as nn_creator
from denspp.offline import get_path_to_project
from denspp.offline.plot_helper import save_figure, get_plot_color, get_textsize_paper
from denspp.offline.analog.common_func import CommonDigitalFunctions


if __name__ == "__main__":
    # --- Settings
    val_max = 4.
    total_bits = 5

    # --- Definitions
    stimulus_full = np.linspace(start=-val_max, stop=+val_max, num=201, endpoint=True)
    yrange = (float(stimulus_full.min()), float(stimulus_full.max()))
    frac_bits = total_bits-int(np.log2(val_max))-1

    func = CommonDigitalFunctions()
    func.define_limits(True, total_bits, frac_bits)
    stimulus_qnt = func.quantize_fxp(stimulus_full)

    stim_full = Tensor(stimulus_full)
    stim_qnt = Tensor(stimulus_qnt)

    idx = -1
    while True:
        idx += 1
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(7, 5))
        match idx:
            case 0:
                act0 = nn_torch.ReLU()
                out0 = act0(Tensor(stim_full))
                act1 = nn_creator.ReLU(total_bits)
                out1 = act1(Tensor(stim_qnt))

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full')
                axs.step(stim_qnt, out1, color=get_plot_color(1), label=f'FxP({total_bits}, {frac_bits})', where="mid")

                axs.set_title('ReLU', fontsize=get_textsize_paper())
                axs.set_yticks([0., 1., 2., 3., 4.])
                axs.set_ylim([-0.02, val_max+0.02])
            case 1:
                num_list = [5, 21]
                act0 = nn_torch.Tanh()
                out0 = act0(stim_full)
                act1 = nn_creator.Tanh(total_bits, frac_bits, num_list[0])
                out1 = act1(stim_qnt)
                act2 = nn_creator.Tanh(total_bits, frac_bits, num_list[1])
                out2 = act2(stim_qnt)

                axs.plot(stimulus_full, out0, color=get_plot_color(0), label='Full')
                axs.step(stim_qnt, out1, color=get_plot_color(1),
                         label=f'FxP({total_bits}, {frac_bits}), num={num_list[0]}', where="mid")
                axs.step(stim_qnt, out2, color=get_plot_color(2),
                         label=f'FxP({total_bits}, {frac_bits}), num={num_list[1]}', where="mid")

                axs.set_title('Tanh', fontsize=get_textsize_paper())
                axs.set_yticks([-1., -.5, 0., .5, 1.])
                axs.set_ylim([-1.02, +1.02])
            case 2:
                act0 = nn_torch.Tanh()
                out0 = act0(stim_full)
                act1 = nn_torch.Hardtanh()
                out1 = act1(stim_full)
                act2 = nn_creator.HardTanh(total_bits, frac_bits)
                out2 = act2(stim_qnt)

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full (Tanh)')
                axs.plot(stim_full, out1, color=get_plot_color(1), label=f'Full (HardTanh)')
                axs.step(stim_qnt, out2, color=get_plot_color(2),
                         label=f'FxP({total_bits}, {frac_bits})', where="mid")

                axs.set_title('HardTanh', fontsize=get_textsize_paper())
                axs.set_yticks([-1., -.5, 0., .5, 1.])
                axs.set_ylim([-1.02, +1.02])
            case 3:
                num_list = [3, 8]
                act0 = nn_torch.Sigmoid()
                out0 = act0(stim_full)
                act1 = nn_creator.Sigmoid(total_bits, frac_bits, num_list[0])
                out1 = act1(stim_qnt)
                act2 = nn_creator.Sigmoid(total_bits, frac_bits, num_list[1])
                out2 = act2(stim_qnt)

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full')
                axs.step(stim_qnt, out1, color=get_plot_color(1),
                         label=f'FxP({total_bits}, {frac_bits}), num={num_list[0]}', where="mid")
                axs.step(stim_qnt, out2, color=get_plot_color(2),
                         label=f'FxP({total_bits}, {frac_bits}), num={num_list[1]}', where="mid")

                axs.set_title('Sigmoid', fontsize=get_textsize_paper())
                axs.set_yticks([0., .25, .5, .75, 1.])
                axs.set_ylim([-0.02, +1.02])
            case 4:
                act0 = nn_torch.Sigmoid()
                out0 = act0(stim_full)
                act1 = nn_torch.Hardsigmoid()
                out1 = act1(stim_full)
                act2 = nn_creator.HardSigmoid(total_bits, frac_bits)
                out2 = act2(stim_qnt)

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full (Sigmoid)')
                axs.plot(stim_full, out1, color=get_plot_color(1), label=f'Full (HardSigmoid)')
                axs.step(stim_qnt, out2, color=get_plot_color(2),
                         label=f'FxP({total_bits}, {frac_bits})', where="mid")

                axs.set_title('HardSigmoid', fontsize=get_textsize_paper())
                axs.set_yticks([0., .25, 0.5, .75, 1.])
                axs.set_ylim([-0.02, +1.02])
            case 5:
                out0 = absolute(stim_full)
                out1 = absolute(stim_qnt)

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full')
                axs.step(stim_qnt, out1, color=get_plot_color(1), label=f'FxP({total_bits}, {frac_bits})', where='mid')

                axs.set_title('Absolute', fontsize=get_textsize_paper())
                axs.set_yticks([0., 1., 2., 3., 4.])
                axs.set_ylim([-0.02, +4.02])
            case 6:
                out0 = stim_full
                out1 = heaviside(stim_qnt, zeros_like(stim_full))
                out2 = sign(stim_qnt)

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Input')
                axs.step(stim_qnt, out1, color=get_plot_color(1), label=f'Heaviside, FxP({total_bits}, {frac_bits})', where='mid')
                axs.step(stim_qnt, out2, color=get_plot_color(2), label=f'Sign, FxP({total_bits}, {frac_bits})', where='mid')

                axs.set_title('Heaviside / Sign', fontsize=get_textsize_paper())
                axs.set_yticks([-1., -.5, 0., 5., 1.])
                axs.set_ylim([-1.02, +1.02])
            case 7:
                act0 = nn_torch.PReLU(init=0.125)
                out0 = act0(Tensor(stim_full)).detach().numpy()
                act1 = nn_torch.PReLU(init=0.125)
                out1 = act1(Tensor(stim_qnt)).detach().numpy()

                axs.plot(stim_full, out0, color=get_plot_color(0), label='Full')
                axs.step(stim_qnt, out1, color=get_plot_color(1), label=f'FxP({total_bits}, {frac_bits})', where="mid")

                axs.set_title('PReLU (a=0.125)', fontsize=get_textsize_paper())
                axs.set_yticks([0., 1., 2., 3., 4.])
                axs.set_ylim([-0.52, val_max + 0.02])
            case _:
                break

        axs.grid()
        axs.legend(loc='upper left', fontsize=get_textsize_paper())
        axs.set_xlim(yrange)
        axs.set_xticks([yrange[0], yrange[0] / 2, 0., yrange[1] / 2, yrange[1]])
        axs.tick_params(axis='both', labelsize=get_textsize_paper()-3)
        axs.set_ylabel('Output Value', fontsize=get_textsize_paper())
        axs.set_xlabel('Input Value', fontsize=get_textsize_paper())

        plt.tight_layout()
        save_figure(fig, path=f"{get_path_to_project()}/runs", name=f"quant_act{idx:02d}")
    plt.show()
