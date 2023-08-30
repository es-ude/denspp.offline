import numpy as np
import torch

if __name__ == "__main__":
    path2model = "runs/20230531_164911_train_dnn_dae_v1/model_369"
    # path2model = "runs/20230830_162608_train_dnn_ae_v1/model_474"

    model = torch.load(path2model)

    print("Output of weights in AI model")

    ite = 0
    for name, param in model.named_parameters():
        print("--------------------")
        print(f"\nIteration i={ite}")
        print(name)
        A = param
        print(param)

    print("TEST")