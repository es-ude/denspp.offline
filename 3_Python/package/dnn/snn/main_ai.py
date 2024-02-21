# imports
#import snntorch as snn
#from snntorch import surrogate
#from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
#from snntorch import spikeplot as splt
import package.dnn.snn.pytorch_snn_control as Network
import package.dnn.snn.net_utility as nutil

import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
#import torch.nn.functional as F

import package.dnn.snn.plot_snn as show

import matplotlib.pyplot as plt
from scipy.io import savemat
#from scipy.io import loadmat
#import snntorch.spikeplot as splt


#import numpy as np
#from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
#import itertools

import os
import time
#from pprint import pprint

#import torchvision
#import torchaudio
#import torchinfo
#import tqdm


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# TODO: Move setings in seperate file
#load_module = False

#path2mat = "data/SpAIke/denoising_dataset_File1_Sorted.mat"  # Datensatz enthält Fehler
path2mat = "data/Martinez_2009/2023-04-17_Dataset01_SimDaten_Martinez2009_Sorted"


# variabeles for Dataloader
load_dataset = 3  # 1: mnist, 2: SpAIke, 3: time Variant SpAIke,
batch = [16,32,64,128,256, 512] #[1,2,4,8,16,32,64,128,256]
subset = 10  # reduce size of mnist by factor
valid_size = 0.1  # create validationset size out of dataset
exclude_list = []  # exclude Labels from SpAIke dataset
dt = 2e-6  # minimal time diverence between two spikes
sample_rate = 20e3  # sample rait of input data

# learning parameter
num_steps = 5  # time steps to hold input data
num_epoch = 60  # maximal learn epochs

# early stoping parameter
early_stop = False  # use early stop function
patience = 5  # early stop after "patience" times validation loss hat not impruft
early_stop_delta = 0  # toleranc for decision if thomshing wos learnd
restore_weight = True

# optimiser
optimizer = torch.optim.Adam
#optimizer = torch.optim.SGD
#optimizer = torch.optim.RMSprop

#loss Function
#loss_fn = SF.ce_temporal_loss()
#loss_fn = SF.mse_temporal_loss()
loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)#, population_code=True, num_classes=5)  # time var etwa 70%
#loss_fn = SF.ce_count_loss(population_code=True, num_classes=5)  # after all time steps / time var etwa
#loss_fn = SF.ce_rate_loss()  # for each time step

# spikes per sample
#hold_input = int(1/(sample_rate*dt))  # holdtime/ neuron_min_action_time
hold_input = [256]
num_sample = 3
seed = 0


def main(steps, batch_size, path):
    print("load_model...")
    name = f'Kombinierter ausgang_{steps}steps'

    net = Network.NeuronalNetwork(name, device='cpu', device_warn=False, path=path)  # device='cpu'

    match load_dataset:  # 1: mnist, 2: SpAIke 3: SpAIke timebased, defold none
        case 1:
            print("load mnist dataset")
            net.load_mnist(batch_size=batch_size, subset=subset, valid_size=valid_size)
            net.set_time_invariant(num_steps=steps)
        case 2:
            print("load Spaike dataset")
            net.load_SpAIke(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
                            exclude_list=exclude_list)
            net.set_time_invariant(num_steps=steps)
        case 3:
            print("load time variant Spaike")
            net.load_SpAIke_time(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
                                 exclude_list=exclude_list, hold_data=steps,sample_rate=sample_rate, dt=dt)  # hold_input
            net.set_time_variant()
        case _:
            print("\033[91mNot implemented dataset\033[0m")
            return

    print("set options for learning process...")
    net.set_optimiser_loss(optimizer=optimizer,
                           loss_fn=loss_fn)
    net.set_early_stop(early_stop=early_stop, patience=patience,
                       early_stop_delta=early_stop_delta, restore_weight=restore_weight)
    print("----------------------------------------------------------------")
    print("==========================================================================================")
    print(f'Modelname: {net.get_name()}')
    #net.print_summary()
    print("==========================================================================================")

    print("----------------------------------------------------------------")
    print("train_model...")

    start_time = time.time()
    net.model_tran(num_epoch=num_epoch)
    end_time = time.time()

    program_time = end_time - start_time
    print(f'execution time: {time.strftime("%H h,%M min, %S s", time.gmtime(program_time))}')

    print("save module...")
    utils.reset(net.module)
    nutil.save_net(net, net.get_path())


    # show loss
    show.loss_acc_plot(net=net, path=net.get_path()+"figures/", save_name="acc_loss_plot", show=False)

    # show one output
    utils.reset(net.module)
    spk, target = net.generate_spk_target_data(net.valid_loader)
    #print('targets in batch')

    #print(f'target {target}')
    show.output_raster(spk[:, 0], net.get_path()+"figures/", f'spk_out_target_{net.label[target[0]]}', show=False)
    utils.reset(net.module)
    spk3, spk2, spk1, spk0, spk, target, sig=net.generate_out_with_hidden_spikes()
    #print(spk.size())
    show.output_raster(spk3[:, 0, :], net.get_path() + "figures/", f'spk3_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk2[:, 0, :], net.get_path() + "figures/", f'spk2_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk1[:, 0, :], net.get_path() + "figures/", f'spk1_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk0[:, 0, :], net.get_path() + "figures/", f'spk0_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk[:, 0, :], net.get_path() + "figures/", f'spk_hidden_target_{net.label[target[0]]}', show=False)
    #print(f'input signal size {sig.size()}')
    #print(spk[:, 0])
    acc_t, loss_t = net.get_train_acc_loss()
    acc_v, loss_v = net.get_train_acc_loss()


    mdict = {"acc_training": acc_t, "loss_training": loss_t,"acc_validation": acc_v, "loss_validation": loss_v,
             "output": spk3[:, 0, :].tolist(), "third_layer": spk2[:, 0, :].tolist(), "second_layer":spk1[:, 0, :].tolist(),
             "first_leyer":spk0[:, 0, :].tolist(),"steps_persampel":steps,"Batch_size": batch_size,"sampels": 32,
             "endoder_layer_output":spk[:, 0, :].tolist(), "Tarcet_class":target[0].tolist(), "input_signal":sig[0].tolist()}

    savemat(net.get_path() + "../"+ name + ".mat", mdict)
    show.plot_SpAIke_combined(spk[:, 0,:], sig[0], target, 1*32,path=net.get_path() + "figures/",
                              save_name=f'spk_encoder_target_{net.label[target[0]]}')
    show.plot_SpAIke_combined(spk3[:, 0, :], sig[0], target, 1 * 32, path=net.get_path() + "figures/",
                              save_name=f'spk_in_out_target_{net.label[target[0]]}')
    print()
    print("data generation für combination")
    spk, targets, inputs =net.combine_input_and_gen_out(3)

    print(targets)
    show.plot_SpAIke_combined(spk[:, 0], inputs, targets, 256, path=net.get_path() + "figures/",
                              save_name=f'spk_combinaton_target_{net.label[targets[0]]}_{net.label[targets[1]]}_{net.label[targets[2]]}')

    #show.plot_SpAIke_combined_with_winner_take_all(spk[:, 0], inputs, targets, 256, path=net.get_path() + "figures/",
    #                          save_name=f'spk_combinaton_target_{net.label[targets[0]]}_{net.label[targets[1]]}_{net.label[targets[2]]}')

    #plt.show()
    plt.close('all')
    ret_acc = net.valid_acc_rec[:]
    ret_loss = net.valid_loss_rec[:]
    del net

    return ret_acc, ret_loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    acc=[]
    loss=[]
    path="run/"
    name = "batchweepX"
    num_loops = len(hold_input)

    for i in range(num_loops):
        acc_i, loss_i = main(hold_input[i]/32,256,path)
        acc.append(acc_i)
        loss.append(loss_i)

    # save data as .mat
    acc = np.array(acc)
    loss = np.array(loss)

    PROJECT_PATH = os.path.abspath("")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)
    mdict = {"acc": acc, "loss": loss}
    savemat(path+name+".mat", mdict)

    # plot data
    fig, ax = plt.subplots(ncols=2, figsize=(8, 6),
                           sharex=True)  # gridspec_kw = {'height_ratios': [0.2, 1, 0.6]}
    for i in range(len(acc)):
        ax[0].plot(acc[i])  # label="Train Accuracy"
    ax[0].set_title("Accuracy")
    ax[0].legend(batch)

    for i in range(len(loss)):
        ax[1].plot(loss[i])

    ax[1].set_title("loss")
    ax[1].legend(batch)

    plt.xlabel("Epoch")

    plt.savefig(path + name + ".png", dpi=300, bbox_inches='tight')
    plt.close('all')
