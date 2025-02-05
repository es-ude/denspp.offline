from fractions import Fraction
import numpy as np
from scipy import integrate as it


def do_quantize_transient(transient_orig: dict, fs_new: float, u_lsb: float, i_gain: float=2e3) -> dict:
    """Performing a re-quantization of the transient input signal (amplitude and time)
    Args:
        transient_orig:     Input dictionary with transient signal ['V': voltage, 'I': current, 'fs': sampling rate]
        fs_new:             New sampling rate
        u_lsb:              New smallest voltage resolution (least significant bit, LSB)
    Returns:
        Dictionary with new transient output ['V': voltage, 'I': current, 'fs': sampling rate]
    """
    current0 = do_resample_time(transient_orig['I'], transient_orig['fs'], fs_new, do_offset_comp=True)
    voltage0 = do_resample_time(transient_orig['V'], transient_orig['fs'], fs_new)

    current_tran = do_resample_amplitude(i_gain * current0, u_lsb) / i_gain
    voltage_tran = do_resample_amplitude(voltage0, u_lsb)
    return {'I': current_tran, 'V': voltage_tran, 'fs': fs_new}


def do_resample_time(signal_in: np.ndarray, fs_orig: float, fs_new: float,
                     do_offset_comp: bool=False) -> np.ndarray:
    """Do resampling of time value from transient signals
    Args:
        signal_in:      Numpy array of transient input signal
        fs_orig:        Original sampling rate value
        fs_new:         New sampling rate value
        do_offset_comp: Do offset compensation on output
    Returns:
        Numpy array of resampled into
    """
    from scipy.signal import resample_poly

    u_chck = np.mean(signal_in)
    u_off = u_chck if not do_offset_comp else 0
    if not fs_orig == fs_new:
        p, q = Fraction(fs_new / fs_orig).limit_denominator(10000).as_integer_ratio()
        return u_off + resample_poly(signal_in - u_chck, p, q)
    else:
        return signal_in - u_off


def do_resample_amplitude(signal_in: np.ndarray, u_lsb: float) -> np.ndarray:
    """Do resampling of amplitude from transient signal
    Args:
        signal_in:  Numpy array with transient signal
        u_lsb:      New smallest voltage resolution (least significant bit, LSB)
    Returns:
        Numpy array with re-sampled input (amplitude)
    """
    return u_lsb * np.round(signal_in / u_lsb, 0) if not u_lsb == 0.0 else signal_in


def calculate_charge_injected(i_in: np.ndarray, fs: float) -> np.ndarray:
    """Calculating the injected charge amount of one stimulation pattern"""
    time = np.linspace(0, i_in.size, num=i_in.size) / fs
    return it.cumtrapz(i_in, time, dx=1/fs, initial=0)
