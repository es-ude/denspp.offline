import torch
import numpy as np


class DataNormalization:
    """Normalizing the input data to enhance classification performance.

    Args:
        device (str):   Defining the resource/device for processing ["CPU", "GPU", or "FPGA"]
        method (str):   The normalization method ["minmax", "binary", "norm", "zscore", "medianmad", or "meanmad"]
        mode (str):     Defining the normalization mode ['None', 'bipolar', 'global', 'combined']
    Methods:
        normalize(): Normalize the input data based on the selected mode and method.
    Examples:
        # Create an instance of DataNormalization
        data_normalizer = DataNormalization(frames_in, mode="GPU", method="minmax", do_bipolar=True, do_global=False)

        # Normalize the data
        normalized_frames = data_normalizer.normalize(frames_in: np.ndarray)
            frames_in: Input data to be normalized.
    """
    _do_bipolar: bool
    _do_global: bool

    def __init__(self, device: str, method: str, mode: str):
        self.mode = device
        self.method = method
        self._define_mode(mode)
        self.__do_hw_mode = {'CPU': self._normalize_cpu, 'GPU': self._normalize_gpu, 'FPGA': self._normalize_fpga}

    def _define_mode(self, type: str) -> None:
        """"""
        match type:
            case 'bipolar':
                do_bipolar = True
                do_global = False
            case 'global':
                do_bipolar = False
                do_global = True
            case 'combined':
                do_bipolar = True
                do_global = True
            case _:
                do_bipolar = False
                do_global = False
        self._do_bipolar = do_bipolar
        self._do_global = do_global

    def _normalize_cpu(self, frames_in: np.ndarray):
        mean_val = 0 if self._do_bipolar else 0.5
        scale_mean = 1 if self._do_bipolar else 2
        scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if self._do_global else 1

        std_global = np.std(frames_in) if self._do_global else 1
        mean_global = np.mean(frames_in) if self._do_global else 1
        median_global = np.median(frames_in) if self._do_global else 1
        mad_global = np.median(np.absolute(frames_in - np.median(frames_in))) if self._do_global else 1

        frames_out = np.zeros(shape=frames_in.shape)

        match self.method:
            case "minmax":
                for i, frame in enumerate(frames_in):
                    scale_local = np.max([np.max(frame), -np.min(frame)]) if not self._do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    frames_out[i, :] = mean_val + frame / scale

            case "binary":
                for i, frame in enumerate(frames_in):
                    scale_local = np.max([np.max(frame), -np.min(frame)]) if not self._do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    division_value = 0
                    while scale > (2 ** division_value):
                        division_value += 1
                    frames_out[i, :] = mean_val + frame / (2 ** division_value)

            case "norm":
                for i, frame in enumerate(frames_in):
                    scale = np.linalg.norm(frame)
                    frames_out[i, :] = frame / scale

            case "zscore":
                for i, frame in enumerate(frames_in):
                    std_local = np.std(frame) if not self._do_global else 1
                    mean_local = np.mean(frame) if not self._do_global else 1
                    mean = mean_local * mean_global
                    std = std_local * std_global
                    frames_out[i, :] = (frame - mean) / std

            case "medianmad":
                for i, frame in enumerate(frames_in):
                    median_local = np.median(frame) if not self._do_global else 1
                    mad_local = np.median(np.absolute(frame - np.median(frame))) if not self._do_global else 1
                    median = median_local * median_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - median) / mad

            case "meanmad":
                for i, frame in enumerate(frames_in):
                    mean_local = np.mean(frame) if not self._do_global else 1
                    mad_local = np.mean(np.absolute(frame - np.mean(frame))) if not self._do_global else 1
                    mean = mean_local * mean_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - mean) / mad

        return frames_out

    def _normalize_gpu(self, frames_in: torch.Tensor):
        mean_val = 0 if self._do_bipolar else 0.5
        scale_mean = 1 if self._do_bipolar else 2
        scale_global = torch.max(torch.abs(frames_in)).item() if self._do_global else 1

        std_global = torch.std(frames_in).item() if self._do_global else 1
        mean_global = torch.mean(frames_in).item() if self._do_global else 1
        median_global = torch.median(frames_in).item() if self._do_global else 1
        mad_global = torch.median(
            torch.abs(frames_in - torch.median(frames_in))).item() if self._do_global else 1

        frames_out = torch.zeros_like(frames_in)

        match self.method:
            case "minmax":
                for i, frame in enumerate(frames_in):
                    scale_local = torch.max(torch.abs(frame)).item() if not self._do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    frames_out[i, :] = mean_val + frame / scale

            case "binary":
                for i, frame in enumerate(frames_in):
                    scale_local = torch.max(torch.abs(frame)).item() if not self._do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    division_value = 0
                    while scale > (2 ** division_value):
                        division_value += 1
                    frames_out[i, :] = mean_val + frame / (2 ** division_value)

            case "norm":
                for i, frame in enumerate(frames_in):
                    scale = torch.norm(frame)
                    frames_out[i, :] = frame / scale

            case "zscore":
                for i, frame in enumerate(frames_in):
                    std_local = torch.std(frame) if not self._do_global else 1
                    mean_local = torch.mean(frame) if not self._do_global else 1
                    mean = mean_local * mean_global
                    std = std_local * std_global
                    frames_out[i, :] = (frame - mean) / std

            case "medianmad":
                for i, frame in enumerate(frames_in):
                    median_local = torch.median(frame) if not self._do_global else 1
                    mad_local = torch.median(torch.abs(frame - torch.median(frame))) if not self._do_global else 1
                    median = median_local * median_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - median) / mad

            case "meanmad":
                for i, frame in enumerate(frames_in):
                    mean_local = torch.mean(frame) if not self._do_global else 1
                    mad_local = torch.mean(torch.abs(frame - torch.mean(frame))) if not self._do_global else 1
                    mean = mean_local * mean_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - mean) / mad

        return frames_out

    def _normalize_fpga(self, frames_in: np.ndarray, simple_method=False) -> np.ndarray:
        mean_val = 0 if self._do_bipolar else 0.5
        scale_mean = 1 if self._do_bipolar else 2
        scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if self._do_global else 1

        frames_out = np.zeros(shape=frames_in.shape)

        for i, frame in enumerate(frames_in):
            scale_local = np.max([np.max(frame), -np.min(frame)]) if not self._do_global else 1
            scale = scale_mean * scale_local * scale_global
            division_value = 1

            while scale > (2 ** division_value):
                division_value += 1

            maximum = scale_global if self._do_global else scale_local
            adjust_maximum = maximum
            divider = 2**division_value if self._do_bipolar else 2 ** (division_value - 1)
            coeff = [0, 0, 0, 0]
            for j in range(1, 5):
                if adjust_maximum + adjust_maximum / (2 ** j) <= divider:
                    adjust_maximum = adjust_maximum + adjust_maximum / (2**j)
                    coeff[j - 1] = 1
            if simple_method:
                frames_out[i, :] = mean_val + frame / (2 ** division_value)
            else:
                frames_out[i, :] = mean_val
                frames_out[i, :] += frame / (2 ** division_value)
                frames_out[i, :] += coeff[0] * frame / (2 ** (division_value + 1))
                frames_out[i, :] += coeff[1] * frame / (2 ** (division_value + 2))
                frames_out[i, :] += coeff[2] * frame / (2 ** (division_value + 3))
                frames_out[i, :] += coeff[3] * frame / (2 ** (division_value + 4))
        return frames_out

    def normalize(self, frames_in: np.ndarray) -> np.ndarray:
        """Do normalisation of data
        Args:
            Numpy array with frames for normalizing
        Returns:
            Numpy array with normalized frames
        """
        if self.mode in self.__do_hw_mode.keys():
            return self.__do_hw_mode[self.mode](frames_in)
        else:
            print("Selected mode is not available.")
            return np.array(0.0)
