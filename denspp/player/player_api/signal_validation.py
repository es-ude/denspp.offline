import numpy as np
from scipy import signal
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SignalValidationResult:
    """Dataclass to store results of signal validation
    Attributes:
        pearson_correlation (list[float]): Pearson correlation coefficient between original and processed signals
        mean_squared_error (list[float]): Mean Squared Error between original and processed signals
        spectral_coherence (list[float]): Spectral coherence between original and processed signals
    """
    pearson_correlation: list[float] = None 
    mean_squared_error: list[float] = None
    spectral_coherence: list[float] = None  

class SignalCompartor:
    _signal_original_with_cut: np.ndarray # original signal, cuted to the processed signal length
    _signal_processed: np.ndarray # processed signal
    _fs_original: int # original sampling frequency
    _fs_processed: int # processed sampling frequency
    _scaling_factor: float # scaling factor
    _equal_number_of_signals: bool # whether the original and processed signals have equal length
    _results: SignalValidationResult # results of the signal validation


    def __init__(self, original_data_with_cut: np.ndarray, signal_processed: np.ndarray, fs_original: int, fs_processed: int, scaling_factor: float) -> None:
        self._signal_original_with_cut = original_data_with_cut
        self._signal_processed = signal_processed
        self._fs_original = fs_original
        self._fs_processed = fs_processed
        self._scaling_factor = scaling_factor
        self._equal_number_of_signals = self.check_equal_number_of_signals
        self._signal_original_with_cut = original_data_with_cut
        self._results = SignalValidationResult()
        
    
    def analyze_signals(self) -> None:
        """Analyze the original and processed signals"""
        self._results.pearson_correlation = self._calculate_pearson_korrelation()
        self._results.mean_squared_error = self._calculate_mse()
        self._results.spectral_coherence = self._calculate_spectral_coherence()

    @property
    def get_results(self) -> SignalValidationResult:
        """Get the results of the signal validation

        Returns:
            SignalValidationResult: Results of the signal validation
        """        
        return self._results

    @property
    def check_equal_number_of_signals(self) -> bool:
        """Check if the total length of the original and processed signals are equal

        Returns:
            bool: True if equal, False otherwise
        """        
        return len(self._signal_original_with_cut) == len(self._signal_processed)
        

    def _reverse_vertical_scaling(self) -> np.ndarray:
        """Reverse the vertical scaling applied to the processed signal

        Returns:
            np.ndarray: Processed signal after reversing vertical scaling
        """        
        return self._signal_processed * self._scaling_factor
    

    def _calculate_pearson_korrelation(self) -> list:
        """Calculate Pearson correlation coefficient between original and processed signals

        Raises:
            ValueError: If the original and processed signals do not have the same amount of channels

        Returns:
            list: List of Pearson correlation coefficients for each signal pair
        """               
        results =[]
        if not self._equal_number_of_signals:
            raise ValueError("Original and processed signals must have the same length for similarity calculation.")
        for i, _ in enumerate(self._signal_original_with_cut):
            if len(self._signal_original_with_cut[i]) != len(self._signal_processed[i]):
                x_orig = np.linspace(0, 1, len(self._signal_original_with_cut[i])) #x Values for original signal
                x_proc = np.linspace(0, 1, len(self._signal_processed[i])) # x Values for processed signal

                processed_aligned = np.interp(x_orig, x_proc, self._signal_processed[i]) # Align processed signal to original signal length, e.g. Samplingrate
            else:
                processed_aligned = self._signal_processed[i]
            
            correlation_matrix = np.corrcoef(self._signal_original_with_cut[i], processed_aligned)
            pearson_r = correlation_matrix[0, 1]
            results.append(pearson_r)
        return results
    
    def _calculate_mse(self) -> list:
        """Calculate Mean Squared Error (MSE) between original and processed signals

        Raises:
            ValueError: If the original and processed signals do not have the same amount of channels

        Returns:
            list: List of Mean Squared Errors for each Channel comparison
        """        
        results =[]
        if not self._equal_number_of_signals:
            raise ValueError("Original and processed signals must have the same length for similarity calculation.")
        for i, _ in enumerate(self._signal_original_with_cut):
            if len(self._signal_original_with_cut[i]) != len(self._signal_processed[i]):
                x_orig = np.linspace(0, 1, len(self._signal_original_with_cut[i]))
                x_proc = np.linspace(0, 1, len(self._signal_processed[i]))

                processed_aligned = np.interp(x_orig, x_proc, self._signal_processed[i]) ## Align processed signal to original signal length, e.g. Samplingrate
            else:
                processed_aligned = self._signal_processed[i]


            orig_norm = (self._signal_original_with_cut[i] - np.min(self._signal_original_with_cut[i])) / (np.max(self._signal_original_with_cut[i]) - np.min(self._signal_original_with_cut[i]))
            proc_norm = (processed_aligned - np.min(processed_aligned)) / (np.max(processed_aligned) - np.min(processed_aligned))
            mse = np.mean((orig_norm - proc_norm) ** 2)
            results.append(mse)
        return results
    
    def _calculate_spectral_coherence(self, self_defined_nperseg =None) -> list:
        results =[]
        if not self._equal_number_of_signals:
            raise ValueError("Original and processed signals must have the same length for similarity calculation.")
        for i, _ in enumerate(self._signal_original_with_cut):
            if len(self._signal_original_with_cut[i]) != len(self._signal_processed[i]):
                x_orig = np.linspace(0, 1, len(self._signal_original_with_cut[i]))
                x_proc = np.linspace(0, 1, len(self._signal_processed[i]))
                processed_aligned = np.interp(x_orig, x_proc, self._signal_processed[i]) ## Align processed signal to original signal length, e.g. Samplingrate
            else:
                processed_aligned = self._signal_processed[i]
            
            if self_defined_nperseg is None: # define nperseg; default is 1/4 of the signal length 
                nperseg = len(processed_aligned) // 4
            else:
                nperseg = self_defined_nperseg
            
            f, Cxy = signal.coherence(self._signal_original_with_cut[i], processed_aligned, fs=self._fs_original, nperseg=nperseg)
            self._plot_spectral_coherence(f, Cxy)
            self._show_signal_plots(self._signal_original_with_cut[i], processed_aligned)
            
            results.append({
                "frequencies": f,
                "coherence_values": Cxy,
                "max_coherence": np.max(Cxy),
            })
        return results


    def _plot_spectral_coherence(self, f, Cxy) -> None:
        """Plot the spectral coherence between original and processed signals"""
        plt.semilogy(f, Cxy)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.title('Spectral Coherence between Original and Processed Signals')
        plt.grid()
        plt.show()


    def _show_signal_plots(self, data1, data2) -> None:
        """Show plots of the original and processed signals for visual inspection"""
        plt.plot(data1, color='k', marker='.', markersize=4)
        plt.plot(data2, color='b', marker='.', markersize=4)

        #plt.xlim([data.timestamps[0], data.timestamps[-1]])
        plt.xlabel("Timestamp")
        plt.ylabel(f"ADC output V]")

        plt.tight_layout()
        plt.show()