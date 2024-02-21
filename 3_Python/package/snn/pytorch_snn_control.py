import itertools
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim.lr_scheduler as lr_s
from torchinfo import summary
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import utils
from snntorch import backprop
from snntorch import functional as SF

from scipy import interpolate
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import scr_ai.dae_dataset as dae
import scr_ai.net_utility as nutil
from scr_ai.snn_architecture import Net_h as Net
import scr.plotting as show


class NeuronalNetwork(Net):
    def __init__(self, model_name, device=False, device_warn=True, path="model/"):
        """create a Network-Class with all nesesary variabeles"""
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

        self.model_mode = "USE"
        self.date_time=datetime.now().strftime('%Y%m%d_%H%M%S')
        self._model_path = self._check_path(path + self.date_time+"/")
        self._data_path = self._check_path("data/")
        self._mnist_path = self._check_path("data/mnist")
        self._model_name = self.date_time + "_" + model_name

        self.torch_version = torch.__version__
        self.snntorch_version = snn.__version__
        print(f'Pytorch version = {self.torch_version}')
        print(f'Snntorch version = {self.snntorch_version}')

        self.load_model = False

        # for loading of data and dataloader
        self.dtype = torch.float
        self.data_train = None
        self.target_train = None
        self.data_valid = None
        self.target_valid = None
        self.label = []
        self.fraim_mean = None

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
        self.time_first = True

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

        # create a Copie of the model (trainingsmodel: module / beckupmodel: module0
        self.module0 = self.module
        utils.reset(self.module)

    def _check_path(self, path):
        """check if path exists and create if nesecery"""
        PROJECT_PATH = os.path.abspath("../scr_ai")
        rel_path = os.path.join(PROJECT_PATH, path)
        if not os.path.exists(rel_path):
            os.makedirs(rel_path)
        return path

    def set_early_stop(self, early_stop=True, patience=5, early_stop_delta=0, restore_weight=False):
        """set parameter for early stopping"""
        self.patience = patience
        self.early_stop = early_stop
        self.early_stop_delta = early_stop_delta
        self.restore_weight = restore_weight

    def model_tran(self, num_epoch):
        """bein training"""
        # options
        self.model_mode = "train"
        self.num_epoch = num_epoch
        self.train_device = self.device

        # learning
        self.module.to(self.device)
        self._train_nn()

    def _train_nn(self):
        """trainingsloop"""
        counter = 0
        es_stop=False
        if self.optimizer == torch.optim.Adam:
            self.optimizer = self.optimizer(self.module.parameters(), lr=1e-2, betas=(0.9, 0.999))
        if self.optimizer == torch.optim.SGD:
            self.optimizer = self.optimizer(self.module.parameters(), lr=1e-2, momentum=0.9)
        if self.optimizer == torch.optim.RMSprop:
            self.optimizer = self.optimizer(self.module.parameters(), lr=1e-2, momentum=0.9)
        scheduler = lr_s.ExponentialLR(optimizer=self.optimizer, gamma=0.9)
        #scheduler = lr_s.StepLR(optimizer, step_size=5, gamma=0.9)

        for epoch in range(self.num_epoch) if not self.early_stop else itertools.count(0):
            self.module.train()
            if self.static_input:
                #TODO: write own training loop for time static data
                loss = backprop.BPTT(self.module, self.train_loader,
                                     optimizer=self.optimizer,
                                     criterion=self.loss_fn,
                                     num_steps=self.num_steps, time_var=False, device=self.device).detach().cpu()#.item()
            else:
                # TODO: write own training LOOP for time variant data
                loss = self.__training_epoch().detach().cpu()

            scheduler.step()

            # get accuracy for validation and Train data
            self.module.eval()
            train_acc = self.data_accuracy(self.train_loader, self.module)
            valid_acc = self.data_accuracy(self.valid_loader, self.module)

            # TODO: change data loader by loss calculation to input tensoren?
            valid_loss = self.data_loss(self.valid_loader, self.module).item()

            # check for early stop if activated
            counter = self.es_check(valid_loss, valid_acc, epoch, counter if self.early_stop else 0)

            # set output for epoch
            print(end="\r")
            print(self._get_status_string(epoch, loss, valid_loss, counter, train_acc, valid_acc))

            # determin and save parameter from network
            self.train_loss_rec.append(loss)
            self.train_acc_rec.append(train_acc)
            self.valid_loss_rec.append(valid_loss)
            self.valid_acc_rec.append(valid_acc)
            self.best_loss_rec.append(self.best_loss)
            self.best_acc_rec.append(self.best_acc)
            self.not_learnd_for.append(counter)

            # abord if condition is reached
            es_stop = self.es_break(counter, epoch)
            if es_stop:
                break
        if not es_stop:
            print(f'reach max number of epoch cycle({self.num_epoch})')
            print(f'restore waights {self.es_restore_waights()}')

    #TODO: create own class for early stop?
    def es_check(self, valid_loss, valid_acc,epoch=0, es_counter =0):
        """ceck for early stopping"""
        counter = es_counter
        if self.early_stop or self.restore_weight:
            # deside if somthing was lerned after epoch 1
            if self.best_loss == None :
                self.best_loss = valid_loss
                self.best_acc = valid_acc
                self.module0 = self.module
                nutil.save_net(self, f'{self.get_path()}ceckpoint/best{epoch}/')
                counter = 0
            elif self.best_loss - valid_loss > self.early_stop_delta:
                self.best_loss = valid_loss
                self.best_acc = valid_acc
                counter = 0
                self.module0 = self.module
                nutil.save_net(self, f'{self.get_path()}ceckpoint/best{epoch}/')
            elif self.best_loss - valid_loss < self.early_stop_delta:
                counter += 1
            return counter
        else:
            return 0

    def es_restore_waights(self):
        """restore best waights from early stopping"""
        if self.restore_weight:
            self.module = self.module0
        return self.restore_weight

    def es_break(self, counter, epoch):
        """check if early stoping has reached its pacience"""
        fired = False
        if self.early_stop:
            if counter == self.patience:
                print(f'reatching of early stop condition at epoch {epoch} ')
                fired = True
        return fired

    def _get_status_string(self, epoch, train_loss, valid_loss, es_counter, train_acc, valid_acc):
        """set stining for status sumary after ech epoch"""
        status = f'epoch: {epoch}' + \
                f'{f"[{((epoch + 1) / self.num_epoch) * 100:.2f}%]" if not self.early_stop else ""}' + \
                f', loss: {train_loss} ' + \
                f'[{valid_loss:.3f}]/' + \
                f'[{(self.best_loss - valid_loss) if self.early_stop else self.valid_loss_rec[-1] - valid_loss if self.valid_loss_rec != [] else 0:.3f}]' + \
                f'{f", counter: {es_counter}/{self.patience}" if self.early_stop else ""}' + \
                f', accuracy: {train_acc:.3f}% ' + \
                f'[{valid_acc:.3f}%]'
        return status

    def valid_nn(self):
        acc = self.data_accuracy(self.valid_loader, self.module)
        print(f'accuracy is: {acc}')
        return acc

    def _forward_pass(self, data):
        """forwordpath"""
        mem_rec = []
        spk_rec = []
        utils.reset(self.module)  # resets hidden states for all LIF neurons in net

        if self.static_input:
            for step in range(self.num_steps):
                spk_out, mem_out = self.module(data)
                spk_rec.append(spk_out)
                mem_rec.append(mem_out)
        else:
            #TODO: move time first to dataloader and have one option
            if self.time_first:
                for step in range(data.size(0)):  # data.size(0) = number of time steps
                    spk_out, mem_out = self.module(data[step])
                    spk_rec.append(spk_out)
                    mem_rec.append(mem_out)
            else:
                for step in range(data.size(1)):  # data.size(0) = number of time steps
                    spk_out, mem_out = self.module(data[:, step, :])
                    spk_rec.append(spk_out)
                    mem_rec.append(mem_out)
        return torch.stack(spk_rec), torch.stack(mem_rec)

    def traind_model2device(self):
        """set model in evaluation mode and send to device """
        self.module.to(self.device)
        self.module.eval()

    def output_time_stat2variant(self, data_loader, num_steps, num_samples, hidden):
        """generate output for multipel static inputs after each other"""
        dat, targets = next(iter(data_loader))
        dat = dat.to(self.device)
        self.module.eval()

        with torch.no_grad():
            mem_rec = []
            spk_rec = []
            mem_rech = []
            spk_rech = []
            ind = []
            t_l = []

            for i in range(num_samples):
                t_l.append(targets[i])#.cpu())
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

    def combine_input_and_gen_out(self, num_sampels):
        """combine multiple inputs ang generate output"""
        value=0
        empty1 = torch.full((256,1),(value), dtype=torch.float32)
        empty2 = torch.full((256,1),(-value), dtype=torch.float32)
        empty = torch.cat((empty1,empty2),1)

        dat, targets = next(iter(self.valid_loader))
        comb_data= dat[0]
        comb_targets = targets[0].unsqueeze(0)
        print(f'new combined data: {comb_data.size()}')
        print(comb_targets.size())
        for i in range(1,num_sampels):
            comb_targets = torch.cat((comb_targets, targets[i].unsqueeze(0)))
            comb_data = torch.cat((comb_data, empty))
            comb_data = torch.cat((comb_data, dat[i]))

        comb_data = comb_data.to(self.device)
        comb_data = comb_data
        utils.reset(self.module)
        with torch.no_grad():
            utils.reset(self.module)
            self.module.to(self.device)
            self.module.eval()
            spk_rec = []
            for step in range(comb_data.size(0)):
                spk_out, mem_out = self.module(comb_data[step].unsqueeze(0))
                spk_rec.append(spk_out)

        return torch.stack(spk_rec).detach().cpu(), comb_targets.detach(), comb_data

    def generate_out_with_hidden_spikes(self):
        """combine multiple inputs ang generate output"""
        dat, targets = next(iter(self.valid_loader))
        dat = dat.to(self.device)
        with torch.no_grad():
            self.module.to(self.device)
            self.module.eval()
            spk_rec3 = []
            spk_rec2 = []
            spk_rec1 = []
            spk_rec0 = []
            spk_rec = []
            utils.reset(self.module)  # resets hidden states for all LIF neurons in net
            for step in range(dat.size(1)):
                spk_out3, mem_out, spk_out2, spk_out1, spk_out0, spk_out = self.module(dat[:,step,:],hidden=True)
                spk_rec.append(spk_out)
                spk_rec0.append(spk_out0)
                spk_rec1.append(spk_out1)
                spk_rec2.append(spk_out2)
                spk_rec3.append(spk_out3)

        return torch.stack(spk_rec3).detach().cpu(), torch.stack(spk_rec2).detach(), torch.stack(spk_rec1).detach(),\
            torch.stack(spk_rec0).detach(), torch.stack(spk_rec).detach(), targets.detach(), dat.detach()

    def data_accuracy(self, data_loader, module):
        """generate accuracy of given dataset"""
        qbar = tqdm(data_loader,desc=f'calculate acuracy ...', leave=False)
        if self.static_input:
            with torch.no_grad():
                total = 0
                acc = 0.0
                self.module.eval()

                for data, targets in qbar:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    spk_rec, _ = self._forward_pass(data)
                    acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                    total += spk_rec.size(1)
                    qbar.set_description(f'calculate acuracy: {(acc / total) * 100:.3f}')
        else:
            with torch.no_grad():
                total = 0
                acc = 0.0
                self.module.eval()

                for data, targets in qbar:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    spk_rec, _ = self._forward_pass(data)
                    accb = SF.accuracy_rate(spk_out=spk_rec, targets=targets)
                    accb = accb * spk_rec.size(1)
                    acc += accb
                    total += spk_rec.size(1)
                    qbar.set_description(f'calculate acuracy: {(acc / total) * 100:.3f}')
        return (acc / total) * 100

    def acc_of_first(self, spk, target, num=0):
        """generate acuracy of first data from batch"""
        size = spk.size()
        spk1 = torch.reshape(spk[:, num, :], (size[0], 1, size[2]))
        print(spk1.size())
        print(spk1.sum(0))
        acc = SF.accuracy_rate(spk1, target[num])#, population_code=True, num_classes=5)
        print(f'acuracy: {acc}')

    def data_loss(self, data_loader, module):
        """calculate loss from given dataset"""
        qbar = tqdm(data_loader,desc=f'calculated loss ...',leave=False)
        with torch.no_grad():
            self.module.eval()
            loss = 0.0
            total_sum = 0
            for data, targets in qbar:
                total_sum += targets.size(0)

                data = data.to(self.device)
                targets = targets.to(self.device)
                spk_rec, _ = self._forward_pass(data)
                loss += self.loss_fn(spk_rec, targets) * targets.size(0)
                qbar.set_description(f'calculated loss: {loss.item() / total_sum}')
        return loss / total_sum

    def load_mnist(self, batch_size, subset=False, path=False, valid_size=0.1, seed=False):
        """load mnist dataset"""
        self.dataset_name = "MNIST"
        self.static_input = True
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = 28*28
        self.num_outputs = 10
        self.batch_size = batch_size
        self.subset = subset
        self.label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        PROJECT_PATH = os.path.abspath("../scr_ai")
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

        mnist_train, mnist_valid = utils.valid_split(ds_train=mnist_train, ds_val=mnist_valid, split=valid_size, seed=seed)

        # Create DataLoaders
        self.train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(mnist_valid, batch_size=batch_size, shuffle=True, drop_last=False)

    def load_SpAIke(self, path, batch_size, subset=False, valid_size=0.1, exclude_list=False):
        """load SpAIke dataset"""
        self.dataset_name = "SpAIke"
        self.input_current = False
        self.ext_spike_gen = False
        self.num_inputs = 40
        self.batch_size = batch_size
        self.subset = subset
        PROJECT_PATH = os.path.abspath("../scr_ai")
        data_path = os.path.join(PROJECT_PATH, path)

        # load data
        annots = loadmat(data_path)
        label_list = [[element for element in upperElement] for upperElement in annots['frames_cluster']]
        con_list = [[[element for element in upperElement]] for upperElement in annots['frames_in']]

        unique, counts = np.unique(label_list, return_counts=True)
        total = sum(counts)
        rel = total/counts

        class_waight = rel/np.min(rel)
        result = np.column_stack((unique, counts))
        print(result)
        print(class_waight)
        self.label = [str(element) for element in list(unique)]
        self.num_outputs = len(unique)

        # exclude outputs
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
        neuron_number = [label.index(element) for element in label_list]

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

        self.train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
        self.data_train = x_train
        self.target_train = y_train

        self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler)
        self.data_valid = x_test
        self.target_valid = y_test

    def load_SpAIke_time(self, path, batch_size, subset=False, valid_size=0.1, exclude_list=False, hold_data=0,
                         dt=1e-6, sample_rate=20e3):
        """load SpAIke dataset and transform it to time veriing data"""

        max_sempel_points = int(1/(sample_rate*dt))   # holdtime/ neuron_min_action_time

        # set parameter
        self.dataset_name = "SpAIke_time"
        self.input_current = False
        self.ext_spike_gen = False
        self.time_first = False
        self.num_inputs = 1
        self.batch_size = batch_size
        self.subset = subset
        PROJECT_PATH = os.path.abspath("../scr_ai")
        data_path = os.path.join(PROJECT_PATH, path)

        # load data
        annots = loadmat(data_path)

        # generates size of: [upperElement, element]
        label_list = [[element for element in upperElement] for upperElement in annots['frames_cluster']]
        if len(label_list) < len(label_list[0]):
            label_list = np.reshape(label_list, (len(label_list[0]), 1))
            label_list = label_list.tolist()
        # generates size of: [upperElement,1, element]
        con_list = [[element for element in upperElement] for upperElement in annots['frames_in']]

        #generate
        print('original data')
        unique, counts = np.unique(label_list, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)

        show.histogram(unique, counts,"row_data",self.get_path()+"figures/")

        x,y,self.fraim_mean = dae.prepare_dae_training(path,True, 2156,[],[0]) # ,7156
        unique, counts = np.unique(y, return_counts=True)

        # generate data new function and aproximate value
        ax, points = np.shape(x)
        if hold_data !=2:
            d = np.zeros([ax, int(points*hold_data)])
            t = np.linspace(0, points * max_sempel_points, points)
            tnew = np.linspace(0, points * max_sempel_points, int(points*hold_data))
            for i in range(ax):
                p=x[i]
                f = interpolate.interp1d(t, p, kind='cubic')
                d[i] = f(tnew)
            x = d

        print(f'neues maximum: {np.max(x)}')
        print(f'neues minimum: {np.min(x)}')
        con_list= x.tolist()
        shape = np.shape(y)
        y = y.reshape((shape[0],1))
        label_list = y.tolist()

        print('after equalisation data')
        unique, counts = np.unique(label_list, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)

        show.histogram(unique, counts, "augmented Data", self.get_path() + "figures/")

        # generate labels and number output_neurons
        self.label = ['C'+str(element) for element in list(unique)]
        self.num_outputs = len(unique)
        print(self.label)

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

        print(f'after exclusion of labels: {exclude_list}')

        unique, counts = np.unique(label_list, return_counts=True)
        result = np.column_stack((unique, counts))
        print(result)

        label_list = torch.tensor(label_list)
        label_list = label_list.squeeze()

        #convert to tensor and normelize data
        con_list = torch.tensor(con_list, dtype=self.dtype)
        con_list = torch.reshape(con_list, (con_list.size(0), con_list.size(1), 1))

        list2 = -con_list
        con_list = torch.cat((con_list, list2), dim=-1)

        # split dataset in data for training and validation (test)
        x_train, x_test, y_train, y_test = train_test_split(con_list, label_list, test_size=valid_size)

        print(f'check data type of train data : {type(x_train)}')
        print(f'check data typeof train label : {type(y_train)}')
        print(f'check data typeof valid data : {type(x_test)}')
        print(f'check data typeof valid label: {type(y_test)}')

        # create datasets
        dataset_train = TensorDataset(x_train, y_train)
        dataset_test = TensorDataset(x_test, y_test)

        # create Dataloader for test and train data
        self.train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

        # save row data after split
        self.data_train = x_train
        self.target_train = y_train
        self.data_valid = x_test
        self.target_valid = y_test

        # controle outputs
        print(f'set validation size x: {x_test.size()} y:{y_test.size()}')
        print(f'set train  size x: {x_train.size()} y:{y_train.size()}')

    def set_optimiser_loss(self, optimizer, loss_fn):
        """set optimiser and loss function"""
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def set_time_invariant(self, num_steps):
        """set dataset to static and give repitition of datapoints"""
        self.static_input = True
        self.num_steps = num_steps

    def set_time_variant(self):
        """set dataset to time veriing"""
        self.static_input = False
        self.num_steps = False

    def set_nn_module(self, module):
        """set the model to a given model"""
        self.module = module
        self.module0 = module

    def generate_output_data(self, data_loader):
        """generate output data from dataset"""
        self.module.to(self.device)
        data, targets = next(iter(data_loader))

        data = data.to(self.device)
        self.module.eval()
        spk_rec, mem_rec = self._forward_pass(data)

        return spk_rec.detach().cpu(), mem_rec.detach().cpu()

    def generate_spk_target_data(self, data_loader):
        """generates spice train and targets from dataset"""
        self.module.eval()
        self.module.to(self.device)
        loader = iter(data_loader)

        data, targets = next(loader)
        size = data.size()
        print(size)
        print(targets.size())
        spk_rec, _ = self._forward_pass(data)

        return spk_rec.detach().cpu(), targets.detach()

    def print_summary(self):
        """generate a summary of model"""
        print(self.dataset_name)
        match self.dataset_name:
            case "SpAIke_time":
                summary(self.module, input_size=(self.batch_size,1))  # input_size=(self.batch_size,1,1)
            case "SpAIke":
                summary(self.module, input_size=(self.batch_size,1,40))  # , device=self.device
            case "MNIST":
                summary(self.module, input_size=(self.batch_size,1,28,28))
            case _:
                print(f"no match fore dataset_name {self.dataset_name}")

    def __training_epoch(self):
        """trainingsrotine per epoch"""

        qbar = tqdm(self.train_loader, desc=f'training ...', leave=False)
        self.module.to(self.device)
        self.module.train()

        for data, targets in qbar:
            spk_rec = []
            loss_rec = 0.
            num_data = 0.
            data = data.to(self.device)
            targets = targets.to(self.device)
            utils.reset(self.module)
            if self.static_input:
                for step in range(self.num_steps):
                    spk, _ =self.module(data)
                    spk_rec.append(spk)  # forward-pass
            else:
                if self.time_first:
                    for step in range(data.size(0)):
                        spk, _ = self.module(data)
                        spk_rec.append(spk)  # forward-pass
                else:
                    for step in range(data.size(1)):
                        spk, _ = self.module(data.transpose(1, 0)[step])
                        spk_rec.append(spk)  # forward-pass
            spk_rec = torch.stack(spk_rec)
            #print(spk_rec.size())
            loss = self.loss_fn(spk_rec, targets)  # loss calculation
            self.optimizer.zero_grad()  # null gradients
            loss.backward()  # calculate gradients
            self.optimizer.step()  # update weights
            loss_rec += loss*spk_rec.size(1)
            num_data += spk_rec.size(1)
            avg_loss = loss_rec/num_data
            qbar.set_description(f'training ... loss per batch: {loss:.3f}, durchschnitsloss: {avg_loss:.3f}')
        return avg_loss

    def get_name(self):
        """get model name"""
        return self._model_name

    def get_path(self):
        """get model path"""
        return self._model_path

    def get_valid_acc_loss(self):
        return self.valid_acc_rec, self.valid_loss_rec

    def get_train_acc_loss(self):
        return self.train_acc_rec, self.train_loss_rec
