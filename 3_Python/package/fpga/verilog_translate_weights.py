import numpy as np
import torch


def read_model_weights(path: str) -> None:
    """Reading DNN model weights for usage in FPGAs"""
    model = torch.load(path)

    print("Output of weights in AI model")

    ite = 0
    for name, param in model.named_parameters():
        print("--------------------")
        print(f"\nIteration i={ite}")
        print(name)
        A = param
        print(param)

    print("TEST 1")