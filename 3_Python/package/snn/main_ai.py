import os
import time
import numpy as np
import torch

from scipy.io import savemat
from snntorch import functional as SF
from snntorch import utils

import package.dnn.snn.pytorch_snn_control as Network
import package.dnn.snn.net_utility as nutil
import package.dnn.snn.plot_snn as show


path2mat = "data/Martinez_2009/2023-04-17_Dataset01_SimDaten_Martinez2009_Sorted"

batch = [16, 32, 64, 128, 256, 512]
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
#loss Function
loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)

# spikes per sample
hold_input = [256]
num_sample = 3
seed = 0


def main(steps: int, batch_size: int, path: str, load_dataset: int):
    """"""
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
    net.set_optimiser_loss(optimizer, loss_fn)
    net.set_early_stop(early_stop, patience, early_stop_delta, restore_weight)
    print("==========================================================================================")
    print(f'Modelname: {net.get_name()}')
    #net.print_summary()
    print("==========================================================================================")
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

    show.output_raster(spk[:, 0], net.get_path()+"figures/", f'spk_out_target_{net.label[target[0]]}', show=False)
    utils.reset(net.module)
    spk3, spk2, spk1, spk0, spk, target, sig=net.generate_out_with_hidden_spikes()

    show.output_raster(spk3[:, 0, :], net.get_path() + "figures/", f'spk3_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk2[:, 0, :], net.get_path() + "figures/", f'spk2_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk1[:, 0, :], net.get_path() + "figures/", f'spk1_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk0[:, 0, :], net.get_path() + "figures/", f'spk0_hidden_target_{net.label[target[0]]}', show=False)
    show.output_raster(spk[:, 0, :], net.get_path() + "figures/", f'spk_hidden_target_{net.label[target[0]]}', show=False)

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
