import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, samplingrate, time, process_step:str):
    """Plot the loaded data for checking purposes"""

    if isinstance(time, list):
        time = time[1] - time[0]
    
    num_points = data.shape[-1] if data.ndim > 1 else data.shape[0]
    duration = num_points / samplingrate
    
    xValue = np.linspace(0, duration, num_points, endpoint=False)

    plt.figure(figsize=(12, 6))
    plt.plot(xValue, data)
    plt.title(f"Data after {process_step}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [v]")
    plt.show()