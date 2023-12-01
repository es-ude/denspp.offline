import matplotlib.pyplot as plt
import numpy as np


def plot_featvec(input: np.ndarray):
    feat = input[:, 0:-2]
    label = input[:, -1]

    plt.figure()
    plt.plot(feat)
    plt.show()