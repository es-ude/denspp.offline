import numpy as np
from scipy.signal import convolve2d


def calculate_reference_car_1d(mea_signal: np.ndarray) -> np.ndarray:
    """Performing the CAR algorithm (Common Average Referencing) on 1D signals"""
    if not len(mea_signal.shape) == 2:
        raise NotImplementedError("The input numpy array has wrong size - Please check!")
    else:
        data_out = np.mean(mea_signal, 0)
        return data_out


def calculate_reference_car_2d(mea_signal: np.ndarray, mapp_used: np.ndarray, kernel_size: int) -> np.ndarray:
    """Performing the CAR algorithm (Common Average Referencing) on 2D signals
    Args:
        mea_signal:     Input signal of transient analysis
        mapp_used:      Overview of active mapping activation function
        kernel_size:    Kernel size for convolution (must be odd-numbered)
    Returns:
        Numpy array with convolved signal for doing common average referencing"""
    if not len(mea_signal.shape) == 3:
        raise NotImplementedError("The input numpy array has wrong size - Please check!")
    else:
        # --- Generating the kernel
        if kernel_size % 2 == 0:
            raise ValueError("Value for building the kernel in CAR algorithm must be odd-numbered!")
        else:
            kernel = np.ones((kernel_size, kernel_size), dtype=float)
            mid_number = int(np.floor(kernel_size / 2))
            kernel[mid_number, mid_number] = 0.0

            kernel = kernel / np.sum(kernel)

        # --- Do the convolution
        data_out = np.zeros(mea_signal.shape, dtype=float)
        for idx in range(0, mea_signal.shape[-1]):
            data_in = mea_signal[:, :, idx]
            conv_out = convolve2d(data_in, kernel, mode='same')
            data_out[:, :, idx] = conv_out

        # --- Correction of not available channels
        for row in range(0, mea_signal.shape[0]):
            for col in range(0, mea_signal.shape[1]):
                if not mapp_used[row, col]:
                    data_out[row, col, :] = np.zeros((mea_signal.shape[-1], ), dtype=float)

        return data_out


###########################################################################
if __name__ == "__main__":
    from src_neuro.call_spike import DataLoader, SettingsDATA

    settings = SettingsDATA(
        path="C:\HomeOffice\Data_Neurosignal",
        data_set=8, data_case=1, data_point=1,
        t_range=[0, 0.5], ch_sel=[], fs_resample=20e3
    )
    data_loader = DataLoader(settings)
    data_loader.do_call()
    data_loader.do_cut()
    data_1d = data_loader.get_data()
    car_1d = calculate_reference_car_1d(data_1d.data_raw)

    data_loader.do_mapping()
    data_2d = data_loader.get_data()
    car_2d = calculate_reference_car_2d(data_2d.data_raw, data_2d.mapping_active, 3)

    del data_loader
