from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import snntorch.nn_snntorch as Network
import snntorch.net_utility as nutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import scr.plotting as show

import matplotlib.pyplot as plt
import numpy as np
import itertools

import os
import time
from pprint import pprint

# optimiser
optimizer = torch.optim.Adam

#loss Function
loss_fn = SF.ce_rate_loss()
#loss_fn = SF.ce_temporal_loss()
#loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
#loss_fn = SF.ce_count_loss()
#loss_fn = SF.mse_membrane_loss()

#parameter for paper
hold_input = 50
num_sample = 3
seed = 0

save_module = True

def main():
    print("load_model...")
    net = Network.NeuronalNetwork('cnn_spaike_time', device_warn=False)

    match load_dataset:  # 1: mnist, 2: SpAIke
        case 1:
            print("load mnist dataset")
            net.load_mnist(batch_size=batch_size, subset=subset, valid_size=valid_size)
            net.set_time_invariant(num_steps=num_steps)
        case 2:
            print("load Spaike dataset")
            net.load_SpAIke(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
                            exclude_list=exclude_list)
            net.set_time_invariant(num_steps=num_steps)
        case 3:
            print("load time variant Spaike")
            net.load_SpAIke_time(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
                            exclude_list=exclude_list)
            net.set_time_variant()
        case _:
            raise Exception("no valid dataset")

    if not load_module:
        print("set options for learning process...")
        net.set_optimiser_loss(optimizer=optimizer,
                               loss_fn=loss_fn)
        net.set_early_stop(early_stop=early_stop, patience=patience,
                           early_stop_delta=early_stop_delta, restore_weight=restore_weight)
        print("----------------------------------------------------------------")
        print(f'Modelname: {net.get_name()}')
        #net.print_summary()
        print("train_model...")
        start_time = time.time()
        net.model_tran(num_epoch=num_epoch)
        end_time = time.time()
        if save_module:
            print("save module...")
            nutil.save_net(net, "model/")
        program_time = end_time - start_time
        print(f'execution time: {time.strftime("%H h,%M min, %S s", time.gmtime(program_time))}')
    else:

        match load_dataset:  # 1: mnist, 2: SpAIke
            case 1:
                print("load mnist module")
                #net = nutil.load_net("./model/", "2023.02.28.13:49_cnn_mnist_plot_paper")
                    # 2023.03.01.08:24_cnn_mnist_plot_paper
                #with Beta learning
                net = nutil.load_net("./model/", "2023.03.01.08:13_cnn_mnist_plot_paper")

            case 2:
                print("load Spaike dataset")
                #net = nutil.load_net("./model/", "2023.02.28.16:57_cnn_SpAIke_plot_paper")

                #with beta learning
                #net = nutil.load_net("./model/", "2023.03.01.07:08_cnn_SpAIke_plot_paper")
                #net = nutil.load_net("./model/", "2023.03.01.13:45_cnn_spaiken_beta_not_learn_self")
                net = nutil.load_net("./model/", "2023.03.01.14:44_cnn_spaiken_beta_not_learn_self")
            case _:
                raise Exception("no valid dataset")

        #net.print_summary()
    # show loss
    # show.loss_acc_nolearn_plot(net=net, show_reset_line=3)
    # show.loss_acc_plot(net=net, show=False)

    # show one output
    # spk, _ = net.generate_output_data(net.valid_loader)
    # spk, target = net.generate_spk_target_data(net.valid_loader)
    # print(spk.size())
    # print(f"target: {net.label[target[0]]}")
    # show.skp_counter_anim(spk[:, 0], spk.size(0), net.label)

    # generate combined input and calculate output
    net.traind_model2device()
    for i in range(1):

        #spk_rec, mem_rec, data_in, data_target = net.output_time_stat2variant(net.train_loader, hold_input, num_sample)
        # print(spk_rec.size())
        # show.skp_counter_anim(spk_rec[:, 0], spk_rec.size(0), net.label)
        name = f'{time.strftime("%Y.%m.%d.%H:%M", time.gmtime(time.time()))}_{net.dataset_name}_exampel_plot'

        # show.skp_counter_anim(spk_rec[:, 0], spk_rec.size(0), net.label)

        match load_dataset:  # 1: mnist, 2: SpAIke
            case 1:
                print("plot mnist dataset")
                spk_rec, mem_rec, data_in, data_target, spk_hidden, mem_hidden = net.output_time_stat2variant(
                    net.valid_loader,
                    hold_input,
                    num_sample,
                    hidden)
                show.plot_mnist_paper(spk_rec[:, 0], data_in, hold_input, num_sample, spk_hidden[:, 0]) # save_name=name
            case 2:
                print("plot Spaike dataset")
                spk_rec, mem_rec, data_in, data_target, spk_hidden, mem_hidden = net.output_time_stat2variant(
                    net.valid_loader,
                    hold_input,
                    num_sample,
                    hidden)
                show.plot_SpAIke_paper(spk_rec[:, 0], data_in, hold_input, num_sample, data_target, spk_hidden[:, 0],
                                       save_name=f"spike_paper_({i})", show=True)
            case 3:
                pass
            case _:
                raise Exception("no valid dataset")

if __name__ == '__main__':
    main()
