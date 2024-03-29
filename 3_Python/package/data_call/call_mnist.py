import os
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset


def load_mnist(data_path: str):
    """Loading MNIST dataset and preparing for training"""

    # --- Checking if dataset exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        do_download = True
    else:
        path2mnist = os.path.join(data_path, 'MNIST')
        if not os.path.exists(path2mnist):
            do_download = True
        else:
            do_download = False

    # --- Resampling of MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    data_train = datasets.MNIST(data_path, train=True, download=do_download, transform=transform)
    data_valid = datasets.MNIST(data_path, train=True, download=do_download, transform=transform)
    data_set = ConcatDataset([data_train, data_valid])
    return data_set
