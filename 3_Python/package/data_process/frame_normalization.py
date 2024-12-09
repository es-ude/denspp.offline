import torch
import numpy as np


class DataNormalization:
    _do_global: bool
    __params: dict = {}
    __extract_peak_mode: int = 2

    def __init__(self, method: str = "minmax", do_global_scaling=False, peak_mode=2):
        """Normalizing the input data to enhance classification performance.
        Args:
            method (str):   The normalization method ["minmax", "norm", "zscore", "medianmad", or "meanmad"]
            do_global_scaling (bool):  Applied global scaling in normalization else sample scaling
        Methods:
            normalize(): Normalize the input data based on the selected mode and method.
        Examples:
            # Create an instance of DataNormalization
            handler = DataNormalization("minmax", "bipolar")

            data_in = (0.5 - np.random.rand(100, 10)) * 10
            normalized_frames = handler.normalize(data_in)

        """
        self.__method = method
        self._do_global = do_global_scaling
        self.__extract_peak_mode = peak_mode
        self.__list_norm_methods = {'zeroone': self._normalize_zeroone, 'minmax': self._normalize_minmax, 'norm': self._normalize_norm,
                                    'zscore': self._normalize_zscore, 'medianmad': self._normalize_medianmad,
                                    'meanmad': self._normalize_medianmad}

    def list_normalization_methods(self) -> None:
        """Printing all avaiable methods for normalization"""
        print(self.__list_norm_methods.keys())

    def get_peak_amplitude_values(self) -> np.ndarray | torch.Tensor:
        """Getting the peak amplitude of rawdata as array"""
        key_search = 'scale_used'
        if key_search in self.__params.keys():
            return self.__params[key_search]
        else:
            raise NotImplementedError("Key scale_local is not available!")

    def normalize(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray:
        """Do normalisation of data
        Args:
            Numpy array with frames for normalizing
        Returns:
            Numpy array with normalized frames
        """
        if self.__method in self.__list_norm_methods.keys():
            return self.__list_norm_methods[self.__method](dataset)
        else:
            raise NotImplementedError("Selected mode is not available.")

    def _generate_tensor_full(self, data: torch.Tensor, num_repeats: int) -> torch.Tensor:
        return torch.repeat_interleave(torch.unsqueeze(data, dim=-1), num_repeats, dim=-1)

    def _generate_numpy_full(self, data: np.ndarray, num_repeats: int) -> np.ndarray:
        return np.repeat(np.expand_dims(data, axis=-1), num_repeats, axis=-1)

    def _get_data_peak_value_numpy(self, raw_dataset: np.ndarray) -> np.ndarray:
        match self.__extract_peak_mode:
            case 0:
                amp_array = np.max(raw_dataset, axis=-1)
            case 1:
                amp_array = np.abs(np.min(raw_dataset, axis=-1))
            case _:
                amp_array = np.max(np.abs(raw_dataset), axis=-1)
        return amp_array

    def _get_data_peak_value_tensor(self, raw_dataset: torch.Tensor) -> torch.Tensor:
        match self.__extract_peak_mode:
            case 0:
                amp_array = torch.max(raw_dataset, dim=-1)
            case 1:
                amp_array = torch.abs(torch.min(raw_dataset, dim=-1).values)
            case _:
                amp_array = torch.max(torch.abs(raw_dataset), dim=-1).values
        return amp_array

    def _get_scaling_value_minmax(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, torch.Tensor):
            scale = torch.max(torch.abs(raw_dataset)) if self._do_global else self._get_data_peak_value_tensor(raw_dataset)
        else:
            scale = np.max(np.abs(raw_dataset)) if self._do_global else self._get_data_peak_value_numpy(raw_dataset)

        self.__params = {'scale_used': scale}

    def _normalize_zeroone(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_minmax(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(2* self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = 0.5 + dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(2* self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = torch.divide(torch.add(0.5, dataset), scale_norm)
        return dataset_norm

    def _normalize_minmax(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_minmax(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = torch.divide(dataset, scale_norm)
        return dataset_norm

    def _get_scaling_value_norm(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if isinstance(raw_dataset, np.ndarray):
            scale = np.linalg.norm(raw_dataset, axis=-1)
        else:
            scale = torch.norm(raw_dataset, dim=-1)
        self.__params = {'scale_used': scale}

    def _normalize_norm(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_norm(dataset)
        if isinstance(dataset, np.ndarray):
            scale_norm = self._generate_numpy_full(self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = dataset / scale_norm
        else:
            scale_norm = self._generate_tensor_full(self.__params['scale_used'], dataset.shape[-1])
            dataset_norm = torch.divide(dataset, scale_norm)
        return dataset_norm

    def _get_scaling_value_zscore(self, raw_dataset: np.ndarray | torch.Tensor) -> None:
        if self._do_global:
            scale_std = np.zeros((raw_dataset.shape[0], )) + np.std(raw_dataset) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0], )) + torch.std(raw_dataset)
            scale_mean = np.zeros((raw_dataset.shape[0], )) + np.mean(raw_dataset) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0], )) + torch.mean(raw_dataset)
        else:
            scale_std = np.std(raw_dataset, axis=-1) if isinstance(raw_dataset, np.ndarray) else torch.std(raw_dataset, dim=-1, unbiased=False)
            scale_mean = np.mean(raw_dataset, axis=-1) if isinstance(raw_dataset, np.ndarray) else torch.mean(raw_dataset, dim=-1)
        self.__params = {'scale_std': scale_std, 'scale_mean': scale_mean}

    def _normalize_zscore(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_zscore(dataset)
        if isinstance(dataset, np.ndarray):
            scale_mean = self._generate_numpy_full(self.__params['scale_mean'], dataset.shape[-1])
            scale_std = self._generate_numpy_full(self.__params['scale_std'], dataset.shape[-1])
            dataset_norm = (dataset - scale_mean) / scale_std
        else:
            scale_mean = self._generate_tensor_full(self.__params['scale_mean'], dataset.shape[-1])
            scale_std = self._generate_tensor_full(self.__params['scale_std'], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_mean), scale_std)

        self.__params['scale_used'] = scale_mean / scale_std
        return dataset_norm

    def _get_scaling_value_medianmad(self, raw_dataset:  np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self._do_global:
            scale_median = np.zeros((raw_dataset.shape[0], )) + np.median(raw_dataset) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0], )) + torch.median(raw_dataset).values
            scale_mad = np.zeros((raw_dataset.shape[0], )) + np.median(np.abs(raw_dataset - np.median(raw_dataset))) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0], )) + torch.median(torch.abs(raw_dataset - torch.median(raw_dataset).values)).values
        else:
            scale_median = np.median(raw_dataset, axis=-1) if isinstance(raw_dataset, np.ndarray) else torch.median(raw_dataset, dim=-1).values
            scale_mad = np.median(np.abs(raw_dataset - self._generate_numpy_full(np.median(raw_dataset, axis=1), raw_dataset.shape[-1])), axis=-1) \
                if isinstance(raw_dataset, np.ndarray) else torch.median(torch.abs(raw_dataset - self._generate_tensor_full(torch.median(raw_dataset, dim=1).values, raw_dataset.shape[-1])), dim=-1).values
        self.__params = {'scale_mad': scale_mad, 'scale_median': scale_median}

    def _normalize_medianmad(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_medianmad(dataset)
        if isinstance(dataset, np.ndarray):
            scale_median = self._generate_numpy_full(self.__params['scale_median'], dataset.shape[-1])
            scale_mad = self._generate_numpy_full(self.__params['scale_mad'], dataset.shape[-1])
            dataset_norm = (dataset - scale_median) / scale_mad
        else:
            scale_median = self._generate_tensor_full(self.__params['scale_median'], dataset.shape[-1])
            scale_mad = self._generate_tensor_full(self.__params['scale_mad'], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_median), scale_mad)

        self.__params['scale_used'] = scale_median / scale_mad
        return dataset_norm

    def _get_scaling_value_meanmad(self, raw_dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self._do_global:
            scale_mean = np.zeros((raw_dataset.shape[0],)) + np.mean(raw_dataset) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0],)) + torch.mean(raw_dataset).values
            scale_mad = np.zeros((raw_dataset.shape[0],)) + np.mean(np.abs(raw_dataset - np.mean(raw_dataset))) \
                if isinstance(raw_dataset, np.ndarray) else torch.zeros((raw_dataset.shape[0],)) + torch.mean(torch.abs(raw_dataset - torch.mean(raw_dataset).values)).values
        else:
            scale_mean = np.mean(raw_dataset, axis=-1) if isinstance(raw_dataset, np.ndarray) else torch.mean(raw_dataset, dim=-1).values
            scale_mad = np.mean(np.abs(raw_dataset - self._generate_numpy_full(np.mean(raw_dataset, axis=1), raw_dataset.shape[-1])), axis=-1) \
                if isinstance(raw_dataset, np.ndarray) else torch.mean(torch.abs(raw_dataset - self._generate_tensor_full(torch.mean(raw_dataset, dim=1), raw_dataset.shape[-1])), dim=-1)
        self.__params = {'scale_mad': scale_mad, 'scale_mean': scale_mean}

    def _normalize_meanmad(self, dataset: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self._get_scaling_value_meanmad(dataset)
        if isinstance(dataset, np.ndarray):
            scale_mean = self._generate_numpy_full(self.__params['scale_mean'], dataset.shape[-1])
            scale_mad = self._generate_numpy_full(self.__params['scale_mad'], dataset.shape[-1])
            dataset_norm = (dataset - scale_mean) / scale_mad
        else:
            scale_mean = self._generate_tensor_full(self.__params['scale_mean'], dataset.shape[-1])
            scale_mad = self._generate_tensor_full(self.__params['scale_mad'], dataset.shape[-1])
            dataset_norm = torch.divide(torch.sub(dataset, scale_mean), scale_mad)

        self.__params['scale_used'] = scale_mean / scale_mad
        return dataset_norm


if __name__ == "__main__":
    def generate_test_data(num_samples: int=100, num_window_size: int=32, do_tensor=False) -> np.ndarray | torch.Tensor:
        if do_tensor:
            x = torch.linspace(0, 2*torch.pi, num_window_size)
            data = torch.zeros((num_samples, num_window_size))
            for idx in range(num_samples):
                data[idx, :] = 10 * torch.sin(x - idx / num_samples * torch.pi)
        else:
            x = np.linspace(0, 2 * np.pi, num_window_size)
            data = np.zeros((num_samples, num_window_size))
            for idx in range(num_samples):
                data[idx, :] = 10 * np.sin(x - idx / num_samples * np.pi)
        return data

    used_method = 'minmax'
    hndl = DataNormalization(used_method)
    hndl.list_normalization_methods()
    num_samples = 100
    num_points = 31

    np_data_in = generate_test_data(num_samples, num_points, False)
    np_data_out = hndl.normalize(np_data_in)
    np_peaks = hndl.get_peak_amplitude_values()

    tr_data_in = generate_test_data(num_samples, num_points, True)
    tr_data_out = hndl.normalize(tr_data_in)
    tr_peaks = hndl.get_peak_amplitude_values()
    del hndl
    print("done")
