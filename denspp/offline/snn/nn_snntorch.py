# imports
# neurales netz
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import torch
import torch.nn as nn
import torch.utils as data
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary

# darstellung und sonstige
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


import numpy as np
import io
import copy
import itertools
import os
import time

# TODO: automatic change of architecture by change of Dataset
# neurale netz architektur
from snn.snn_architecture import Net_h as Net


class NeuronalNetwork(Net):
    def __init__(self, model_name, device=False, device_warn=True):
        super().__init__()

        # generate information about network
        if device:
            self.device = device
        else:
            self.device = "mps" if getattr(torch, 'has_mps', False) else "gpu" if torch.cuda.is_available() else "cpu"
            #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'run on {self.device}')

        if self.device == "mps" and device_warn:
            print('\033[91mWarning: not all functions of SnnTorch Version  are natevly compatibel with mps device\n'
                  'some errors can be fixt with a include of:\n if %%tensor_name%%.is_mps:\n    device= "mps"\033[0m')
        # TODO: change format for bedder readebility and saving
        self.model_mode = "USE"
        self.date_time = time.strftime("%Y.%m.%d.%H:%M", time.gmtime(time.time()))

        self._model_path = self._check_path("model/")
        #self._figure_path = self._check_path("figures/")
        self._data_path = self._check_path("data/")
        self._mnist_path = self._check_path("data/mnist")
        self._model_name = self.date_time + "_" + model_name

        self.load_model = False

        # for loading of data and dataloader
        self.dtype = torch.float
        self.data_train = None
        self.target_train = None
        self.data_valid = None
        self.target_valid = None
        self.label = []

        # information about nn
        self.static_input = True
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = None
        self.num_outputs = None

        # information for training
        self.batch_size = None
        self.num_steps = None
        self.train_device = None
        self.optimizer = None
        self.loss_fn = None
        self.num_epoch = None

        self.patience = None
        self.early_stop = False
        self.early_stop_delta = 0
        self.consecutive_no_learn = True
        self.best_loss = None
        self.best_acc = None
        self.restore_weight = False

        # data_loader
        self.dataset_name = None
        self.train_loader = None
        self.valid_loader = None
        self.exclude_loader = None
        self.subset = 0


        # recordings
        #  hilfsdata
        self.acc_rec = []
        self.loss_rec = []
        self.men_out_rec = []
        self.spk_out_rec = []
        self.target_rec = []

        self.best_loss_rec = []
        self.best_acc_rec = []
        self.not_learnd_for = []

        # data from Training
        self.train_acc_rec = []
        self.train_loss_rec = []

        # data from validation
        self.valid_acc_rec = []
        self.valid_loss_rec = []

        #neuron parameter
        #spike_grad = surrogate.fast_sigmoid(slope=25)

        # create a Copie of the model (trainingsmodel: module / beckupmodel: module0
        self.module = self.module0

    def _check_path(self, path):
        PROJECT_PATH = os.path.abspath(".")
        rel_path = os.path.join(PROJECT_PATH, path)
        if not os.path.exists(rel_path):
            os.makedirs(rel_path)
        return rel_path

    # def load_module(self, model_name: str):
    #     if not os.path.exists(self._model_path + model_name+'.pt'):
    #         raise Exception(f'model does not exist: {self._model_path +  model_name + ".pt"}')
    #     else:
    #         self.load_model = True
    #         net = torch.load(self._model_path+'/'+model_name+'.pt')
    #         self.num_inputs = net.num_inputs
    #         self.num_outputs = net.num_outputs
    #         self.train_device = net.train_device
    #         self.optimizer = net.optimizer
    #         self.loss_fn = net.loss_fn
    #         self.num_epoch = net.num_epoch
    #         self.module = net.module0
    #         self.dataset_name = net.dataset_name
    #         self.valid_loader = net.valid_loader

    # def save_module(self):
    #     if self.device != "cpu":
    #         self.module.cpu()
    #     net = save_object(self.num_inputs, self.num_outputs, self.train_device,
    #                       self.optimizer, self.loss_fn, self.num_epoch, self.module, self.dataset_name, self.date_time, self.valid_loader)
    #     torch.save(net, self._model_path+self._model_name+'.pt')

    def set_early_stop(self, early_stop=True, patience=5, early_stop_delta=0, restore_weight=False):
        self.patience = patience
        self.early_stop = early_stop
        self.early_stop_delta = early_stop_delta
        self.restore_weight = restore_weight



    def model_tran(self, num_epoch):
        # options
        self.model_mode = "train"
        self.num_epoch = num_epoch
        self.train_device = self.device

        # learning
        self.module.to(self.device)
        self._train_nn()


    def _train_nn(self):
        counter = 0
        fired=False
        #div = 1
        for epoch in (range(self.num_epoch)) if not self.early_stop else itertools.count(0):
            self.module.train()
            # lerning proces for epoch:
            if self.static_input:
                loss = backprop.BPTT(self.module, self.train_loader,
                                     optimizer=self.optimizer(self.module.parameters(), lr=1e-2, betas=(0.9, 0.999)),
                                     criterion=self.loss_fn,
                                     num_steps=self.num_steps, time_var=False, device=self.device).item()
            else:
                loss = backprop.BPTT(self.module, self.train_loader,
                                     optimizer=self.optimizer(self.module.parameters(), lr=1e-2, betas=(0.9, 0.999)),
                                     criterion=self.loss_fn,time_first=False,
                                     time_var=True, device=self.device).item()
            valid_acc = self.data_accuracy(self.valid_loader, self.module)
            train_acc = self.data_accuracy(self.train_loader, self.module)
            # TODO: change data loader by loss calculation to input tensoren?
            valid_loss = self.data_loss(self.valid_loader, self.module).item()
            # TODO: auf richtigkeit überprüfen und wiederheerstellung der gewichte überprüfen (module ==module0 -> True)
            if self.early_stop:
                # deside if somthing was lerned after epoch 1
                if self.best_loss == None:
                    self.best_loss = valid_loss
                    self.best_acc = valid_acc
                    self.module0 = self.module
                elif self.best_loss - valid_loss > self.early_stop_delta:
                    self.best_loss = valid_loss
                    self.best_acc = valid_acc
                    counter = 0
                    self.module0 = self.module
                elif self.best_loss - valid_loss < self.early_stop_delta:
                    counter += 1
                    if self.restore_weight:
                        self.module = self.module0
                    valid_acc = self.valid_acc_rec[len(self.valid_acc_rec)-1]
                    train_acc = self.train_acc_rec[len(self.train_acc_rec)-1]
                    loss = self.train_loss_rec[len(self.train_loss_rec)-1]
                    valid_loss = self.valid_loss_rec[len(self.valid_loss_rec)-1]

            print(f'epoch: {epoch }'
                  f'{f"[{((epoch+1) / self.num_epoch) * 100:.2f}%]" if not self.early_stop else ""}'
                  f', loss: {loss:.3f} '
                  f'[{valid_loss:.3f}]/'
                  f'[{self.best_loss - valid_loss if self.early_stop else self.valid_loss_rec[-1]-valid_loss if self.valid_loss_rec != [] else 0:.4f}]'
                  f'{f", counter: {counter}/{self.patience}" if self.early_stop else ""}'
                  f', accuracy: {train_acc:.3f}% '
                  f'[{valid_acc:.3f}%]'
                  )

            # determin parameter from network
            self.train_loss_rec.append(loss)
            self.train_acc_rec.append(train_acc)
            self.valid_loss_rec.append(valid_loss)
            self.valid_acc_rec.append(valid_acc)
            self.best_loss_rec.append(self.best_loss)
            self.best_acc_rec.append(self.best_acc)
            self.not_learnd_for.append(counter)

            # abord if condition is reached
            if self.early_stop:
                if counter == self.patience:
                    print(f'reatching of early stop condition at epoch {epoch} ')
                    fired = True
                    break
        if not fired:
            print(f'reach max number of epoch cycle({self.num_epoch})')


    def valid_nn(self):
        acc = self.data_accuracy(self.valid_loader, self.module)
        print(f'accuracy is: {acc}')
        return acc

    def _forward_pass(self, data):
        mem_rec = []
        spk_rec = []
        utils.reset(self.module)  # resets hidden states for all LIF neurons in net

        if self.static_input:
            for step in range(self.num_steps):
                spk_out, mem_out = self.module(data)
                spk_rec.append(spk_out)
                mem_rec.append(mem_out)
        else:
            for step in range(data.size(0)):  # data.size(0) = number of time steps
                spk_out, mem_out = self.module(data[step])
                spk_rec.append(spk_out)
                mem_rec.append(mem_out)
        return torch.stack(spk_rec), torch.stack(mem_rec)
    def traind_model2device(self):
        self.module.to(self.device)
        self.module.eval()
    def output_time_stat2variant(self, data_loader, num_steps, num_samples, hidden):
        dat, targets = next(iter(data_loader))
        dat = dat.to(self.device)

        with torch.no_grad():
            mem_rec = []
            spk_rec = []
            mem_rech = []
            spk_rech = []
            ind = []
            t_l = []

            for i in range(num_samples):
                t_l.append(targets[i].item())
                print(f'sampel {i+1} : label {self.label[targets[i].detach()]}, neuron number {targets[i].detach()}')
                x = dat[ None, i, :]
                ind.append(x)

            x = torch.stack(ind)
            print(dat[0,:].size())
            print(x[0,0,:].size())
            print(x.size(0))
            utils.reset(self.module)  # resets hidden states for all LIF neurons in net
            for i in range(x.size(0)):
                for step in range(num_steps):  # data.size(0) = number of time steps
                    spk_out, mem_out, spk_h, mem_h = self.module(x[i], hidden=hidden)
                    spk_rec.append(spk_out)
                    mem_rec.append(mem_out)
                    spk_rech.append(spk_h)
                    mem_rech.append(mem_h)

        return torch.stack(spk_rec).detach().cpu(), torch.stack(mem_rec).detach().cpu(), x.detach().cpu(), targets.detach(), \
            torch.stack(spk_rech).detach().cpu(), torch.stack(mem_rech).detach().cpu()

    def data_accuracy(self, data_loader, module):
        self.model_mode = "USE"
        if self.static_input:
            with torch.no_grad():
                total = 0
                acc = 0
                self.module.eval()

                train_loader = iter(data_loader)

                for data, targets in train_loader:

                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    spk_rec, _ = self._forward_pass(data)
                    acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                    total += spk_rec.size(1)
        else:
            with torch.no_grad():
                total = 0
                acc = 0
                self.module.eval()

                train_loader = iter(data_loader)

                for data, targets in train_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    spk_rec, _ = self._forward_pass(data)
                    acc += SF.accuracy_temporal(spk_rec, targets)
                    total += spk_rec.size(1)
        return (acc / total) * 100

    def data_loss(self, data_loader, module):
        self.model_mode= "USE"
        with torch.no_grad():

            loss = 0
            #correct =0
            self.module.eval()

            #train_loader = iter(data_loader)

            # TODO: make generation of data and target tensor not part of function
            data_l = []
            targets_l = []
            for data_batch, targets_batch in data_loader:
                data_l.append(data_batch)
                targets_l.append(targets_batch)


            data = torch.cat(data_l)
            targets = torch.cat(targets_l)

            data = data.to(self.device)
            targets = targets.to(self.device)
            spk_rec, _ = self._forward_pass(data)
            loss = self.loss_fn(spk_rec, targets)

        return loss

    # TODO: make path a input
    def load_mnist(self, batch_size, subset=False, path=False, valid_size=0.1, seed=False):
        self.dataset_name = "MNIST"
        self.static_input = True
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = 28*28
        self.num_outputs = 10
        self.batch_size = batch_size
        self.subset = subset
        self.label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        PROJECT_PATH = os.path.abspath(".")
        if(path):
            data_path = os.path.join(PROJECT_PATH, path)
        else:
            data_path = os.path.join(PROJECT_PATH, self._mnist_path)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_valid = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        #mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)



        if subset:
            # reduce datasets by subsetx to speed up training
            utils.data_subset(mnist_train, subset)
            #utils.data_subset(mnist_test, subset)

        mnist_train, mnist_valid = utils.valid_split(ds_train=mnist_train, ds_val=mnist_valid, split=valid_size, seed=seed)

        # Create DataLoaders
        self.train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(mnist_valid, batch_size=batch_size, shuffle=True, drop_last=False)
        #self.test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    def load_SpAIke(self, path, batch_size, subset=False, valid_size=0.1, exclude_list=False):
        self.dataset_name = "SpAIke"
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = 40
        #self.num_outputs = None # spezified after output
        self.batch_size = batch_size
        self.subset = subset
        PROJECT_PATH = os.path.abspath(".")
        data_path = os.path.join(PROJECT_PATH, path)



        # load data
        annots = loadmat(data_path)
        # generates size of: [upperElement, element]
        label_list = [[element for element in upperElement] for upperElement in annots['frames_cluster']]
        # generates size of: [upperElement,1, element]
        con_list = [[[element.item() for element in upperElement]] for upperElement in annots['frames_in']]

        unique, counts = np.unique(label_list, return_counts=True)
        total = sum(counts)
        rel = total/counts

        #class_waight = 1/counts
        class_waight = rel/np.min(rel)
        result = np.column_stack((unique, counts))
        print(result)
        print(class_waight)
        self.label = [str(element) for element in list(unique)]
        self.num_outputs = len(unique)

        # exclude outputs
        exclude = []
        if exclude_list:
            for excl in exclude_list:
                exclude = []
                for index, element in enumerate(label_list):
                    if element == excl:
                        # print(f'{i},{j}')
                        exclude.append(index)
                for i in reversed(exclude):
                    label_list.pop(i)
                    con_list.pop(i)

        # bestimmung der outputs
        label = [[element] for element in list(unique)]
        #result = np.column_stack((unique, counts))
        #print(result)
        neuron_number = [label.index(element) for element in label_list]
        #print(label_list)
        #print(neuron_number)


        x_train, x_test, y_train, y_test = train_test_split(con_list, neuron_number, test_size=valid_size)

        x_train = torch.tensor(x_train, dtype=self.dtype)
        y_train = torch.tensor(y_train)
        y_train = y_train.flatten()

        x_test = torch.tensor(x_test, dtype=self.dtype)
        y_test = torch.tensor(y_test)
        y_test = y_test.flatten()


        dataset_train = TensorDataset(x_train, y_train)
        dataset_test = TensorDataset(x_test, y_test)


        train_waight= []
        test_waight = []

        for idx, (data, target) in enumerate(dataset_train):
            train_waight.append(class_waight[target])

        train_sampler = WeightedRandomSampler(train_waight, num_samples=len(train_waight), replacement=True)

        for idx, (data, target) in enumerate(dataset_test):
            test_waight.append(class_waight[target])

        test_sampler = WeightedRandomSampler(test_waight, num_samples=len(train_waight), replacement=True)

        # TODO: subset function
        # if subset:
        #     # reduce datasets by subsetx to speed up training
        #     utils.data_subset(dataset_test, subset)
        #     utils.data_subset(dataset_train, subset)

        self.train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
        #self.train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        self.data_train = x_train
        self.target_train = y_train



        self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler)
        #self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        self.data_valid = x_test
        self.target_valid = y_test

    def load_SpAIke_time(self, path, batch_size, subset=False, valid_size=0.1, exclude_list=False):
        self.dataset_name = "SpAIke"
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = 1
        # self.num_outputs = None # spezified after output
        self.batch_size = batch_size
        self.subset = subset
        PROJECT_PATH = os.path.abspath(".")
        data_path = os.path.join(PROJECT_PATH, path)

        # load data
        annots = loadmat(data_path)
        # generates size of: [upperElement, element]
        label_list = [[element for element in upperElement] for upperElement in annots['frames_cluster']]
        # generates size of: [upperElement,1, element]
        con_list = [[element.item() for element in upperElement] for upperElement in annots['frames_in']]

        unique, counts = np.unique(label_list, return_counts=True)
        total = sum(counts)
        rel = total / counts

        # class_waight = 1/counts
        class_waight = rel / np.min(rel)
        result = np.column_stack((unique, counts))
        print(result)
        print(class_waight)
        self.label = [str(element) for element in list(unique)]
        self.num_outputs = len(unique)

        # exclude outputs
        exclude = []
        if exclude_list:
            for excl in exclude_list:
                exclude = []
                for index, element in enumerate(label_list):
                    if element == excl:
                        # print(f'{i},{j}')
                        exclude.append(index)
                for i in reversed(exclude):
                    label_list.pop(i)
                    con_list.pop(i)

        # bestimmung der outputs
        label = [[element] for element in list(unique)]
        # result = np.column_stack((unique, counts))
        # print(result)
        neuron_number = [label.index(element) for element in label_list]
        # print(label_list)
        # print(neuron_number)

        x_train, x_test, y_train, y_test = train_test_split(con_list, neuron_number, test_size=valid_size)
        x_train = torch.tensor(x_train, dtype=self.dtype)
        x_train = np.reshape(x_train, (x_train.size(0),40, 1))
        y_train = torch.tensor(y_train)
        #y_train = y_train.flatten()

        x_test = torch.tensor(x_test, dtype=self.dtype)
        x_test = np.reshape(x_test, (x_test.size(0), 40, 1))
        y_test = torch.tensor(y_test)
        #y_test = y_test.flatten()

        dataset_train = TensorDataset(x_train, y_train)
        dataset_test = TensorDataset(x_test, y_test)

        train_waight = []
        test_waight = []

        for idx, (data, target) in enumerate(dataset_train):
            train_waight.append(class_waight[target])

        train_sampler = WeightedRandomSampler(train_waight, num_samples=len(train_waight), replacement=True)

        for idx, (data, target) in enumerate(dataset_test):
            test_waight.append(class_waight[target])

        test_sampler = WeightedRandomSampler(test_waight, num_samples=len(train_waight), replacement=True)

        # TODO: subset function
        # if subset:
        #     # reduce datasets by subsetx to speed up training
        #     utils.data_subset(dataset_test, subset)
        #     utils.data_subset(dataset_train, subset)

        self.train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
        # self.train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        self.data_train = x_train
        self.target_train = y_train

        self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler)
        # self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        self.data_valid = x_test
        self.target_valid = y_test
        print(x_test.size())
        print(y_test.size())



    # TODO: prüfen ob noch gebraucht
    def load_mat(self, path="data/SpAIke/denoising_dataset_File1_Sorted.mat"):
        PROJECT_PATH = os.path.abspath(".")
        data_path = os.path.join(PROJECT_PATH, path)

        annots = loadmat(data_path)
        label_list = [[element for element in upperElement] for upperElement in annots['frames_cluster']]
        con_list = [[element for element in upperElement] for upperElement in annots['frames_in']]

        unique, counts = np.unique(label_list, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)
        excl_list = []
        for excl in excl_list:
            exclude = []
            for index, element in enumerate(label_list):
                if element == excl:
                    #print(f'{i},{j}')
                    exclude.append(index)
            for i in reversed(exclude):
                label_list.pop(i)
                con_list.pop(i)

        unique, counts = np.unique(label_list, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)

        x = torch.tensor(con_list, dtype=self.dtype)
        y = torch.tensor(label_list, dtype=self.dtype)
        y = y.flatten()

        print(x.size())
        print(y.size())

    def set_optimiser_loss(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def set_time_invariant(self, num_steps):
        self.static_input = True
        self.num_steps = num_steps

    def set_time_variant(self):
        self.static_input = False
        self.num_steps = False

    def set_nn_module(self, module):
        self.module = module

    def generate_output_data(self, data_loader):
        self.module.to(self.device)
        data, targets = next(iter(data_loader))

        data = data.to(self.device)
        self.module.eval
        spk_rec, mem_rec = self._forward_pass(data)

        return spk_rec.detach().cpu(), mem_rec.detach().cpu()

    def generate_spk_target_data(self, data_loader):
        self.module.to(self.device)
        data, targets = next(iter(data_loader))

        data = data.to(self.device)
        self.module.eval
        spk_rec, mem_rec = self._forward_pass(data)

        return spk_rec.detach().cpu(), targets.detach()

    def print_summary(self):
        summary(self.module, self.input_size)

    def get_name(self):
        return self._model_name




# class save_object(nn.Module):
#     def __init__(self, num_inputs, num_outputs, train_device, optimizer, loss_fn, num_epoch, module, dataset_name, date,valid_loader):
#         super().__init__()
#         self.date = date
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.train_device = train_device
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.num_epoch = num_epoch
#         self.dataset_name = dataset_name
#         self.module0 = module
#         self.valid_loader = valid_loader

