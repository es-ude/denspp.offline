import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, samplingrate, time, process_step:str):
    """Plot the loaded data for checking purposes"""

    if isinstance(time, list):
        time = time[1] - time[0]
    plt.figure(figsize=(12, 6))
    dataPoints = int(samplingrate * time)
    xValue = np.linspace(0, time, dataPoints, endpoint=False)
    plt.plot(xValue, data)
    plt.title(f"Data after {process_step}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [v]")
    plt.show()