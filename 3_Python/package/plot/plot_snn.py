import os
import numpy as np
import torch
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

figure_size = () # 16:9


def spk_in_anim(spk_data, save_name=False, show=True, path=False):
    """
    :param spk_data: singe spiketrain to be plottet over time [T x D]
    :param save_name: name to save the plot to, if not providet the plot will not be plotted
    :return:
    """
    if not path:
        path="figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    fig, ax = plt.subplots()
    anim = splt.animator(spk_data, fig, ax)

    if save_name:
        anim.save(path+save_name+".mp4")


def skp_counter_anim(spk_data, num_steps=False, labels=False, save_name=False, show=True, path=False):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    if not labels:
        labels=[]

    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    #labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #  Plot and save spike count histogram
    if not num_steps:
        num_steps = spk_data.size(0)

    splt.spike_count(spk_data, fig, ax, labels, num_steps=num_steps, time_step=1e-3)

    if save_name:
       plt.savefig(path+save_name+".png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    # Animate and save spike count histogram
    if labels:
        anim = splt.spike_count(spk_data, fig, ax, labels, animate=True, interpolate=5, num_steps=num_steps,
                                time_step=1e-3)
    else:
        anim = splt.spike_count(spk_data, fig, ax, animate=True, interpolate=5, num_steps=num_steps,
                            time_step=1e-3)

    if save_name:
        anim.save(path+save_name+".mp4")


def skp_counter( spk_data, num_steps=False, labels=False,titel=False, save_name=False, show=True, path=False):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    if not labels:
        labels=[]

    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    #labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if titel:
        plt.title = titel

    #  Plot and save spike count histogram
    if not num_steps:
        num_steps = spk_data.size(0)

    splt.spike_count(spk_data, fig, ax, labels, num_steps=num_steps, time_step=1e-3)
    if save_name:
       plt.savefig(path+save_name+".png", dpi=300, bbox_inches='tight')


def neuron_traces(data_rec, dim):
    splt.traces(data_rec, dim=dim)


def loss_acc_plot(net, save_name=False, show=True, path=False):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)
    fig, ax = plt.subplots(ncols=2, figsize=(8,6),
                           sharex=True)  # gridspec_kw = {'height_ratios': [0.2, 1, 0.6]}

    ax[0].plot(net.train_acc_rec)  # label="Train Accuracy"
    ax[0].plot(net.valid_acc_rec) # , label="valid Accuracy"
    #ax[0].plot(net.best_acc_rec)
    ax[0].set_title("Accuracy")
    ax[0].legend(["Train Accuracy", "valid Accuracy"])
    #ax[0].set_ylabel("Accuracy")

    ax[1].plot(net.train_loss_rec)
    ax[1].plot(net.valid_loss_rec)
    #ax[1].plot(net.best_loss_rec)
    ax[1].set_title("loss")
    ax[1].legend(["Train loss", "valid loss"])
    #ax[1].set_ylabel("loss")

    plt.xlabel("Epoch")

    if save_name:
        plt.savefig(path + save_name + ".png", dpi=300, bbox_inches='tight')


def loss_acc_nolearn_plot(net, show_reset_line=False, reset_value=False, save_name=False, show=True, path=False):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)
    fig, ax = plt.subplots(ncols=3, figsize=(16, 9),
                           sharex=True)  # gridspec_kw = {'height_ratios': [0.2, 1, 0.6]}

    ax[0].plot(net.train_acc_rec)  # label="Train Accuracy"
    ax[0].plot(net.valid_acc_rec) # , label="valid Accuracy"
    ax[0].plot(net.best_acc_rec)
    ax[0].set_title("Accuracy")
    ax[0].legend(["Train Accuracy", "valid Accuracy", "accuracy at best loss"])
    # ax[0].set_ylabel("Accuracy")

    ax[1].plot(net.train_loss_rec)
    ax[1].plot(net.valid_loss_rec)
    ax[1].plot(net.best_loss_rec)
    ax[1].set_title("loss")
    ax[1].legend(["Train loss", "valid loss", "best loss"])
    # ax[1].set_ylabel("loss")

    ax[2].plot(net.not_learnd_for)
    if show_reset_line and net.early_stop:
        ax[2].axhline(y=net.patience, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    elif show_reset_line and reset_value:
        ax[2].axhline(y=reset_value, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    ax[2].set_title("early stop counter")
    ax[2].legend(["eary stop counter"])
    # ax[2].set_ylabel("loss")

    plt.xlabel("Epoch")

    if save_name:
        plt.savefig(path + save_name + ".png", dpi=300, bbox_inches='tight')


def plot_SpAIke_paper(spk_out, input, num_steps, num_sempele,target, spk_hidd, save_name=False, show=True ,path=False):

    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    #plt.figure(figsize=(16,9), facecolor='w') # , dpi=300

    print(input.size())

    ax10 = plt.subplot2grid((8,3),(0,0), colspan=1)
    ax10.plot(input[0,0,0])
    ax10.set_xticks([0, 20, 40])
    ax10.set_ylabel("input \n Value")
    ax10.set_xlabel("input neuron")
    ax10.axes.yaxis.set_visible(False)
    ax10.title.set_text(f'Target neuron: {target[0]}')

    ax11 = plt.subplot2grid((8, 3), (0, 1), colspan=1)
    ax11.plot(input[1, 0, 0])
    ax11.set_xticks([0, 20, 40])
    ax11.axes.yaxis.set_visible(False)
    ax11.title.set_text(f'Target neuron: {target[1]}')

    ax12 = plt.subplot2grid((8, 3), (0, 2), colspan=1)
    ax12.plot(input[2, 0, 0])
    ax12.set_xticks([0, 20, 40])
    ax12.axes.yaxis.set_visible(False)
    ax12.title.set_text(f'Target neuron: {target[2]}')

    #fig, ax = plt.subplots(nrows=3,sharex=True)  # figsize=(8,6), gridspec_kw = {'height_ratios': [0.2, 1, 0.6]}
    ax2 = plt.subplot2grid((8,3), (2,0), colspan=3, rowspan=2)
    splt.raster(spk_out, ax2, c="black", marker="|")  # s=400, marker="|"
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.axes.xaxis.set_visible(False)
    ax2.set_ylim([-0.5, 5])
    ax2.set_ylabel("Output\nneuron")

    if spk_hidd != []:
        ax3 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=4, sharex=ax2)
        splt.raster(spk_hidd, ax3, c="black", s=.5)  # s=400,
        ax3.set_ylabel("hidden Layer\nneuron")
        ax3.set_xlabel("number steps")

    print(spk_out.size())
    for i in range(num_sempele+1):
        ax2.axvline(x=i*num_steps, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax3.axvline(x=i * num_steps, alpha=0.25, linestyle="dashed", c="black", linewidth=2)

    if save_name:
        print(f'saving {save_name}')
        plt.savefig(path + save_name + ".png", figsize=(16,9), dpi=300)



def plot_mnist_paper(spk_out, input, num_steps, num_sempele, spk_hidd, save_name=False, show=True ,path=False):
    spk_out1 = spk_out
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    print(input.size())

    # plt.figure(figsize=(160,90), dpi=300, facecolor='w')

    ax10 = plt.subplot2grid((6,3),(0,0), colspan=1)
    ax10.imshow(input[0].mean(axis=0).reshape((28, -1)).cpu(), cmap='binary')
    ax10.set_xticks([0, 14, 28])
    ax10.set_yticks([0, 14, 28])
    ax10.set_ylabel("input \n Value")
    ax10.set_xlabel("input neuron")
    #ax10.axes.xaxis.set_visible(False)
    #ax10.axes.yaxis.set_visible(False)
    #ax10.set_ylim([0,28])
    #ax10.set_xlim([0, 28])

    ax11 = plt.subplot2grid((6, 3), (0, 1), colspan=1)
    ax11.imshow(input[1].mean(axis=0).reshape((28, -1)).cpu(), cmap='binary')
    ax11.set_xticks([0, 14, 28])
    ax11.set_yticks([0, 14, 28])
    #ax11.axes.xaxis.set_visible(False)
    #ax11.axes.yaxis.set_visible(False)
    #ax10.set_ylim([0, 28])
    #ax10.set_xlim([0, 28])

    ax12 = plt.subplot2grid((6, 3), (0, 2), colspan=1)
    ax12.imshow(input[2].mean(axis=0).reshape((28, -1)).cpu(), cmap='binary')
    ax12.set_xticks([0, 14, 28])
    ax12.set_yticks([0, 14, 28])
    #ax12.axes.xaxis.set_visible(False)
    #ax12.axes.yaxis.set_visible(False)
    #ax10.set_ylim([0, 28])
    #ax10.set_xlim([0, 28])

    #fig, ax = plt.subplots(nrows=3,sharex=True)  # figsize=(8,6), gridspec_kw = {'height_ratios': [0.2, 1, 0.6]}
    ax2 = plt.subplot2grid((6, 3), (2, 0), colspan=3, rowspan=2)
    splt.raster(spk_out, ax2, c="black", marker="|")  # s=400, marker="|"
    ax2.set_yticks([0, 1, 2, 3, 4,5,6,7,8,9])
    ax2.axes.xaxis.set_visible(False)
    ax2.set_ylim([-0.5, 9.5])
    ax2.set_ylabel("Output\nneuron")

    if spk_hidd!=[]:
        ax3 = plt.subplot2grid((6, 3), (4, 0), colspan=3, rowspan=2, sharex=ax2)
        splt.raster(spk_hidd, ax3, c="black", s=.5)  # s=400,
        ax3.set_ylabel("hidden Layer\nneuron")
        ax3.set_xlabel("number steps")

    for i in range(num_sempele+1):
        ax2.axvline(x=i * num_steps, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax3.axvline(x=i * num_steps, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    ax2.set_ylabel("neuron \nindex ")

    if save_name:
        print(f'saving {save_name}')
        plt.savefig(path + save_name + ".png", figsize=(16,9), dpi=300)


def output_raster(spike_data_sample, path="", save_name="", show = True):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)

    #  s: size of scatter points; c: color of scatter points
    splt.raster(spike_data_sample, ax, s=1.5, c="black")
    plt.title("Output Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    if save_name:
        #print(f'saving {save_name}')
        plt.savefig(path + save_name + ".png", dpi=300, bbox_inches='tight')


def teil_linien_test_plot():
    '''
    Color parts of a line based on its properties, e.g., slope.
    '''

    x = np.linspace(0, 3 * np.pi, 500)
    y = np.sin(x)
    z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

    # Create a colormap for red, green and blue and a norm to color
    # f' < -0.5 red, f' > 0.5 blue, and the rest green
    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(3)

    fig1 = plt.figure()
    plt.gca().add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(-1.1, 1.1)


def plot_SpAIke_combined(spk_out, sig_in, targets, num_sempele, save_name=False, show=True ,path=False):

    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    #plt.figure(figsize=(16,9), facecolor='w') # , dpi=300

    ax1 = plt.subplot2grid((5,3), (0,0), colspan=3, rowspan=2)
    ax1.plot(sig_in[:, 0])
    ax1.plot(sig_in[:, 1])
    ax1.legend=(['positive neuron','negative neuron'])
    #ax1.set_xticks([0, 20, 40])
    ax1.set_ylabel("input \n Value")

    #ax1.axes.yaxis.set_visible(False)
    #ax1.title.set_text(f'Target neuron: {target[0]}')

    ax2 = plt.subplot2grid((5,3), (3,0), colspan=3, rowspan=2, sharex=ax1)
    splt.raster(spk_out, ax2, c="black", marker="|")  # s=400, marker="|"
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    #ax2.axes.xaxis.set_visible(False)
    ax2.set_ylim([-0.5, 6])
    ax2.set_ylabel(" Classification \nOutput")
    ax2.set_xlabel("Time point")
    #
    # if spk_hidd != []:
    #     ax3 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=4, sharex=ax2)
    #     splt.raster(spk_hidd, ax3, c="black", s=.5)  # s=400,
    #     ax3.set_ylabel("hidden Layer\nneuron")
    #     ax3.set_xlabel("number steps")
    t = spk_out.size(0)
    steps = t/num_sempele

    # print(spk_out.size())
    for i in range(int(steps)+1):
        ax1.axvline(x=i*num_sempele, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax2.axvline(x=i * num_sempele, alpha=0.25, linestyle="dashed", c="black", linewidth=2)

    plt.tight_layout()
    if save_name:
        plt.savefig(path + save_name + ".png")
        print(f'saving {save_name}')


def plot_SpAIke_combined_with_winner_take_all(spk_out, sig_in, targets, num_sempele, save_name=False, show=True ,path=False):

    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    di= spk_out.size()
    win_rec= []
    counter = torch.zeros((6))
    winner = torch.zeros((6))
    for i in range(di[0]):
        if 0 == (i % num_sempele):
            counter = torch.zeros((6))
        winner= torch.zeros((6))
        counter = counter+ spk_out[i]
        winner[torch.argmax(counter).item()] = 1.0
        win_rec.append(winner)
    win = torch.stack(win_rec)



    #plt.figure(figsize=(16,9), facecolor='w') # , dpi=300

    ax1 = plt.subplot2grid((6,3), (0,0), colspan=3, rowspan=2)
    ax1.plot(sig_in[:, 0])
    ax1.plot(sig_in[:, 1])
    ax1.legend=(['positive neuron','negative neuron'])
    #ax1.set_xticks([0, 20, 40])
    ax1.set_ylabel("U_in / µV")

    #ax1.axes.yaxis.set_visible(False)
    #ax1.title.set_text(f'Target neuron: {target[0]}')

    ax2 = plt.subplot2grid((6,3), (2,0), colspan=3, rowspan=2, sharex=ax1)
    splt.raster(spk_out, ax2, c="black", marker="|")  # s=400, marker="|"
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    #ax2.axes.xaxis.set_visible(False)
    ax2.set_ylim([-0.5, 6])
    ax2.set_ylabel(" Classification \nOutput")

    ax3 = plt.subplot2grid((6, 3), (4, 0), colspan=3, rowspan=2, sharex=ax1)
    splt.raster(win, ax3, c="black", marker="|")  # s=400, marker="|"
    ax3.set_yticks([0, 1, 2, 3, 4, 5])
    # ax2.axes.xaxis.set_visible(False)
    ax3.set_ylim([-0.5, 6])
    ax3.set_ylabel("WTA\nOutput")
    ax3.set_xlabel("Time point")

    #
    # if spk_hidd != []:
    #     ax3 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=4, sharex=ax2)
    #     splt.raster(spk_hidd, ax3, c="black", s=.5)  # s=400,
    #     ax3.set_ylabel("hidden Layer\nneuron")
    #     ax3.set_xlabel("number steps")
    t = spk_out.size(0)
    steps = t/num_sempele

    # print(spk_out.size())
    for i in range(int(steps)+1):
        ax1.axvline(x=i*num_sempele, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax2.axvline(x=i * num_sempele, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax3.axvline(x=i * num_sempele, alpha=0.25, linestyle="dashed", c="black", linewidth=2)

    plt.tight_layout()
    if save_name:
        plt.savefig(path + save_name + ".png")
        print(f'saving {save_name}')


def histogram(x,y, save_name=False, path=False):
    if not path:
        path = "figures/"

    PROJECT_PATH = os.path.abspath("../../appendix/SNN/scr")
    rel_path = os.path.join(PROJECT_PATH, path)
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)
    plt.bar(x,y)
    plt.xlabel('Class')
    plt.ylabel('number samples')

    if save_name:
        plt.savefig(path + save_name + ".png", dpi=300, bbox_inches='tight')
