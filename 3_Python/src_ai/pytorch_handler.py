import os.path, shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src_ai.dae_dataset import calculate_snr

def setup() -> None:
    os_type = os.name

    device0 = "CUDA" if torch.cuda.is_available() else "CPU"
    if device0 == "CUDA":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"... using PyTorch with {device0} device on {os_type}")


def do_training(model: nn.Module, training_loader, validation_loader, optimizer, loss_fn, epochs: int, model_name: str) -> tuple[str, np.ndarray]:
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
            optimizer, loss_fn
        )

        # --- Validation
        # We don't need gradients on to do reporting
        # model.eval()
        model.train(False)
        valid_loss = 0.0
        snr_valid = []

        total_batches = 0
        for vdata in validation_loader:
            data_in = vdata['frame']
            data_out = vdata['mean_frame']

            _, pred_out = model(data_in)
            # Metric calculation
            loss0 = loss_fn(pred_out, data_out)
            valid_loss += loss0

            ypred0 = pred_out.detach().numpy()
            out0 = data_out.detach().numpy()
            snr_valid.append([calculate_snr(ypred0, out0)])
            total_batches += 1

        valid_loss = valid_loss / total_batches

        print(f'... loss of epoch {epoch+1}/{epochs} [{(epoch+1)/epochs*100:.2f} %]: train = {train_loss:.5f}, valid = {valid_loss:.5f}')

        snr_valid = np.array(snr_valid)
        epoch_plot[epoch, 0] = np.min(snr_valid)
        epoch_plot[epoch, 1] = np.mean(snr_valid)
        epoch_plot[epoch, 2] = np.max(snr_valid)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalar('Loss_train', train_loss)
        writer.add_scalar('Loss_valid', valid_loss)
        writer.add_scalars('Training vs. Validation Loss',
            {'mean(SNR)': epoch_plot[epoch, 1], 'min(SNR)': epoch_plot[epoch, 0], 'max(SNR)': epoch_plot[epoch, 2]},
            epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if valid_loss < best_vloss:
            best_vloss = valid_loss
            model_path = os.path.join(path2log, 'model_{}'.format(epoch_number))
            torch.save(model, model_path)

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

def train_one_epoch(model, training_loader, optimizer, loss_fn):
    running_loss = 0.
    total_batches = 0

    # Here, we use enumerate(training_loader) instead of iter(training_loader)
    # so that we can track the batch index and do some intra-epoch reporting
    for data in training_loader:
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
        total_batches += 1

    avg_loss = running_loss / total_batches

    return avg_loss
