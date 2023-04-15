import os.path, shutil
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def do_training(model: nn.Module, training_loader, validation_loader, optimizer, loss_fn, epochs: int, model_name: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_save = 'runs/fashion_trainer_{}'.format(timestamp)
    writer = SummaryWriter(path_save)
    epoch_number = 0
    best_vloss = 1_000_000.

    timestamp_start = datetime.now()
    timestamp_string = timestamp_start.strftime('%H:%M:%S.%f')
    print(f'\nTraining starts on: {timestamp_string}')
    for epoch in range(epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        # --- Training
        model.train(True)
        avg_loss = train_one_epoch(
            model, training_loader,
            optimizer, loss_fn,
            epoch_number, writer
        )

        # --- Validation
        # We don't need gradients on to do reporting
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            data_in = vdata['frame']
            data_out = vdata['mean_frame']

            _, pred_out = model(data_in)
            vloss = loss_fn(pred_out, data_out)
            running_vloss += vloss

        avg_vloss = running_vloss / (i+1)
        print(f'... loss of epoch {epoch+1}/{epochs} [{(epoch+1)/epochs*100: .2f} %]: train = {avg_loss: .5f}, valid = {avg_vloss: .5f}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': avg_loss, 'Validation': avg_vloss},
            epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(path_save, 'model_{}'.format(epoch_number))
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
    path_model = os.path.join('models', model_name + "_" + timestamp)
    shutil.copy(model_path, path_model)

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
