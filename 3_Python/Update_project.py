import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load data
file_path = 'array.pkl'
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Initialize the 3D array
array_3d = np.zeros((8, 8, 20))

# Populate the 3D array
for i in range(8):
    for j in range(8):
        element = loaded_data[i, j]
        if isinstance(element, np.ndarray) and element.shape == (20,):
            array_3d[i, j, :] = element
        else:
            array_3d[i, j, :] = np.zeros(20)

# Custom 2D convolution function with dynamic kernel normalization based on original data
def custom_2d_convolution(data, kernel_size):
    num_x, num_y, num_t = data.shape
    convolved_data = np.zeros_like(data)

    # Create a boolean mask where the original signal is zero
    original_zero_mask = (data == 0)

    for t in range(num_t):
        spatial_slice = data[:, :, t]
        car_data = np.mean(spatial_slice)  # Common Average Reference (CAR)
        car_subtracted_data = spatial_slice - car_data
        convolved_slice = np.zeros_like(spatial_slice)

        for i in range(num_x):
            for j in range(num_y):
                i_min = max(i - (kernel_size // 2), 0)
                i_max = min(i + (kernel_size // 2) + 1, num_x)
                j_min = max(j - (kernel_size // 2), 0)
                j_max = min(j + (kernel_size // 2) + 1, num_y)

                neighborhood = car_subtracted_data[i_min:i_max, j_min:j_max]

                # Create kernel based on non-zero electrodes in the original data's corresponding neighborhood
                original_neighborhood = data[i_min:i_max, j_min:j_max, t]
                non_zero_count = np.count_nonzero(original_neighborhood)

                # Normalize kernel by the number of non-zero electrodes (valid electrodes) from the original data
                if non_zero_count > 0:
                    kernel_slice = np.ones(neighborhood.shape) / non_zero_count
                else:
                    kernel_slice = np.zeros(neighborhood.shape)

                convolved_value = np.sum(neighborhood * kernel_slice)
                convolved_slice[i, j] = convolved_value

                # Debugging print for specific condition
                if t == 2 and i == 0 and j == 0:
                    print(f"Time point {t}, kernel size {kernel_size}x{kernel_size} - Extracted kernel slice:")
                    print(f"Kernel Slice:\n{kernel_slice}")
                    print(f"Neighborhood:\n{neighborhood}")
                    print(f"Original Neighborhood:\n{original_neighborhood}")
                    print(f"Convolved Value: {convolved_value}")

        convolved_data[:, :, t] = convolved_slice

    return convolved_data

# Convolve data using the dynamically adjusted kernels
convolved_noisy_pickle_small = custom_2d_convolution(array_3d, 3)  # Small kernel (3x3)
convolved_noisy_pickle_large = custom_2d_convolution(array_3d, 5)  # Large kernel (5x5)

# Create a boolean mask where the original signal is zero
zero_mask = (array_3d == 0)

# Apply the mask to the convolved data to set values to zero where the original signal is zero
convolved_noisy_pickle_small[zero_mask] = 0
convolved_noisy_pickle_large[zero_mask] = 0

# Plotting functions
def plot_signals(dataset, title):
    num_x, num_y, num_t = dataset.shape
    t = np.linspace(0, 1, num_t)
    fig, axs = plt.subplots(num_x, num_y, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    for i in range(num_x):
        for j in range(num_y):
            axs[i, j].plot(t, dataset[i, j, :])
            axs[i, j].set_title(f'Signal at ({i},{j})', fontsize=10)
            axs[i, j].set_xlabel('Time', fontsize=8)
            axs[i, j].set_ylabel('Amplitude', fontsize=8)
            axs[i, j].tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_signals_comparison(original, convolved_small, convolved_large, title):
    num_x, num_y, num_t = original.shape
    t = np.linspace(0, 1, num_t)
    fig, axs = plt.subplots(num_x, num_y, figsize=(30, 30))
    fig.suptitle(title, fontsize=8)
    for i in range(num_x):
        for j in range(num_y):
            axs[i, j].plot(t, original[i, j, :], label='Original')
            axs[i, j].plot(t, convolved_small[i, j, :], label='Convolved (3x3)')
            axs[i, j].plot(t, convolved_large[i, j, :], label='Convolved (5x5)')
            axs[i, j].set_title(f'Signal at ({i},{j})', fontsize=5)
            axs[i, j].set_xlabel('Time', fontsize=3)
            axs[i, j].set_ylabel('Amplitude', fontsize=3)
            axs[i, j].legend(fontsize=3)
            axs[i, j].tick_params(axis='both', which='major', labelsize=3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot results
plot_signals(array_3d, "Original Noisy Pickle Dataset")
plot_signals(convolved_noisy_pickle_small, "Convolved Noisy Pickle Dataset with Small CAR Kernel")
plot_signals(convolved_noisy_pickle_large, "Convolved Noisy Pickle Dataset with Large CAR Kernel")

plot_signals_comparison(array_3d, convolved_noisy_pickle_small, convolved_noisy_pickle_large,
                        "Original vs Convolved Noisy Pickle Dataset")
