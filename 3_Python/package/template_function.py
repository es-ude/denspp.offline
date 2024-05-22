import numpy as np


def activation_relu(signal: np.ndarray, scale: float, do_scale=True) -> np.ndarray:
    """This function should do an activation layer
    Args:
        signal:     Input array
        scale:      Scaling value
        do_scale:   Should I do the scaling? [Default: True]
    Returns:
        Output array
    """
    u_out = np.zeros(signal.shape)
    xpos = np.argwhere(signal >= 0)
    u_out[xpos] += scale * signal[xpos]
    return u_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Definition of Input
    u_in = np.linspace(-100, 100, 101, endpoint=True)
    u_out = activation_relu(u_in, 0.77)

    # --- Plotting
    plt.figure()
    plt.plot(u_in, 'k', label="input")
    plt.plot(u_out, 'r', label="output")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
