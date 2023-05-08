import torch

path = "runs/20230424_194708_ai_training_dnn_dae_v1/model_475"

A = torch.load(path)
input = torch.tensor(range(0, 32))

print("TEST")