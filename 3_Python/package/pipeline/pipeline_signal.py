class PipelineSignal:
    def __init__(self) -> None:
        """Pipeline signals for saving 1d signal processing

        Params:
            fs_ana - Sampling rate of the analog processing stage [Hz]
            u_in - Electrode Array Input voltage [V]
            u_pre - Output voltage of the pre-amplifier [V]
            u_spk - Output voltage of the bandpass filter for extracting spike activities [V]
            u_lfp - Output voltage of the low-pass filter for extracting LFP activities [V]
            x_adc - ADC output of spike activities []
        """
        # --- Analog processing stage
        self.u_in = None
        self.u_pre = None
        self.u_spk = None
        self.u_lfp = None
        self.fs_ana = 0.0

        # --- Digital processing stage
        self.x_adc = None  # ADC output
        self.x_spk = None  # Output of digital filtering - spike
        self.x_lfp = None  # Output of digital filtering - lfp
        self.x_sda = None  # Output of Spike Detection Algorithm (SDA)
        self.x_thr = None  # Threshold value for SDA
        self.x_pos = None  # Position for generating frames
        self.frames_orig = None  # Original frames after event-detection (larger)
        self.frames_align = None  # Aligned frames to specific method
        self.features = None  # Calculated features of frames
        self.cluster_id = None  # Clustered events
        self.fs_adc = 0.0  # Sampling rate of the ADC incl. oversampling
        self.fs_dig = 0.0  # Processing rate of the digital part

        # --- NSP processing stage
        self.spike_ticks = None  # Spike Ticks
        self.nsp_post = dict()  # Adding some parameters after calculating some neural signal processing methods
