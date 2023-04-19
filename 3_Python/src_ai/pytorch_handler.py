import os.path, shutil
import numpy as np
from datetime import datetime

from src_ai.dae_dataset import calculate_snr

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def do_training(model: nn.Module, training_loader, validation_loader, optimizer, loss_fn, epochs: int, model_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = 'runs'
    folder = '{}_ai_training_'.format(timestamp) + model_name
    path2model = os.path.join(path, folder)
    path2log = os.path.join(path, folder, 'training')
    writer = SummaryWriter(path2log)

    epoch_number = 0
    best_vloss = 1_000_000.

    timestamp_start = datetime.now()
    timestamp_string = timestamp_start.strftime('%H:%M:%S.%f')
    epoch_plot = np.zeros(shape=(epochs, 3))

    print(f'\nTraining starts on: {timestamp_string}')
    for epoch in range(epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        # --- Training
        model.train(True)
        train_loss = train_one_epoch(
            model, training_loader,
            optimizer, loss_fn,
            epoch_number, writer
        )

        # --- Validation
        # We don't need gradients on to do reporting
        model.train(False)
        valid_loss = 0.0
        snr_valid = []
        for i, vdata in enumerate(validation_loader):
            data_in = vdata['frame']
            data_out = vdata['mean_frame']

            _, pred_out = model(data_in)
            # Metric calculation
            loss0 = loss_fn(pred_out, data_out)
            valid_loss += loss0

            ypred0 = pred_out.detach().numpy()
            out0 = data_out.detach().numpy()
            snr_valid.append([calculate_snr(ypred0, out0)])

        valid_loss = valid_loss / (i+1)
        print(f'... loss of epoch {epoch+1}/{epochs} [{(epoch+1)/epochs*100:.2f} %]: train = {train_loss:.5f}, valid = {valid_loss:.5f}')

        snr_valid = np.array(snr_valid)
        epoch_plot[epoch, 0] = np.min(snr_valid)
        epoch_plot[epoch, 1] = np.mean(snr_valid)
        epoch_plot[epoch, 2] = np.max(snr_valid)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': train_loss, 'Validation': valid_loss, 'mean(SNR)': epoch_plot[epoch, 1], 'min(SNR)': epoch_plot[epoch, 0], 'max(SNR)': epoch_plot[epoch, 2]},
            epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if valid_loss < best_vloss:
            best_vloss = valid_loss
            model_path = os.path.join(path2log, 'model_{}'.format(epoch_number))
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    # --- Ausgabe nach Training
    timestamp_end = datetime.now()
    timestamp_string = timestamp_end.strftime('%H:%M:%S.%f')

    diff_time = timestamp_end - timestamp_start
    diff_string = diff_time
    print(f'Training ends on: {timestamp_string}')
    print(f'Training runs: {diff_string}')
    print(f'\nSave best model: {model_path}')
    shutil.copy(model_path, path2model)

    return path2model, epoch_plot

def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of iter(training_loader)
    # so that we can track the batch index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        data_in = data['frame']
        data_out = data['mean_frame']

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        _, pred_out = model(data_in)
        # Compute the loss and its gradients
        loss = loss_fn(pred_out, data_out)
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        no_batch_print = 100
        if i % no_batch_print == (no_batch_print-1):
            # loss per batch
            last_loss = running_loss / no_batch_print
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
