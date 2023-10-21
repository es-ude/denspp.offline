class PipelineSignal:
    def __init__(self, fs_ana: float, fs_adc: float, osr: int) -> None:
        self.u_in = None            # Input voltage
        self.u_pre = None           # Output of pre-amp
        self.u_spk = None           # Output of analogue filtering - spike acitivity
        self.u_lfp = None           # Output of analogue filtering - lfp
        self.u_quant = None         # Quantization error of ADC
        self.x_adc = None           # ADC output
        self.x_spk = None           # Output of digital filtering - spike
        self.x_dly = None           # Delay signal between SDA and frame generation
        self.x_lfp = None           # Output of digital filtering - lfp
        self.x_sda = None           # Output of Spike Detection Algorithm (SDA)
        self.x_thr = None           # Threshold value for SDA
        self.x_pos = None           # Position for generating frames
        self.frames_orig = None     # Original frames after event-detection (larger)
        self.frames_align = None    # Aligned frames to specific method
        self.features = None        # Calculated features of frames
        self.cluster_id = None      # Clustered events
        self.spike_ticks = None     # Spike Ticks
        self.nsp_post = dict()      # Adding some parameters after calculating some neural signal processing methods

        self.fs_ana = fs_ana
        self.fs_adc = osr * fs_adc
        self.fs_dig = fs_adc
