import numpy as np
from src_neuro.sda.sda_pipeline import Pipeline_Digital
from src_neuro.sda.sda_handler import do_single_run, do_method_sweep
from src_neuro.sda.signal_gen import EasyNeuralData_Generator


# --------------- MAIN PROGRAMME
if __name__ == "__main__":
    path2save = "runs"
    # --- Configuration of SDA properties
    use_smoothing = False
    mode_sda = [1] # , 2, 3, 4, 5]
    mode_thr = [2]

    # --- Sweep variables
    spk_firing_rate = np.linspace(10, 100, 11, endpoint=True)
    # spk_firing_rate = [100]
    spk_snr = np.linspace(-18, 12, 31, endpoint=True)
    # spk_snr = [12]
    # --- Configuration of neural input
    spk_num = 300
    spk_amp = -77
    fs = 25e3

    # --- Preparing the experiment
    data = EasyNeuralData_Generator(spk_num, fs)
    pipeline = Pipeline_Digital(fs)
    spk_period = pipeline.sda.frame_length

    do_single_run(pipeline, data, spk_amp, spk_period, spk_firing_rate[0], spk_snr[0], mode_sda[0], mode_thr[0], path2save)
    do_method_sweep(pipeline, data, spk_amp, spk_period, spk_firing_rate, spk_snr, mode_sda, mode_thr, path2save)
    print("\n The End!")
