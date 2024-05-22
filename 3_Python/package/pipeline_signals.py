class invasive_analog:
    def __init__(self) -> None:
        self.u_in = None            # Input voltage
        self.u_pre = None           # Output of pre-amp
        self.u_spk = None           # Output of analogue filtering - spike acitivity
        self.u_lfp = None           # Output of analogue filtering - lfp
        self.x_adc = None           # ADC output
        self.fs_ana = 0.0           # "Sampling rate"


class invasive_digital:
    def __init__(self) -> None:
        self.x_adc = None           # ADC output
        self.x_spk = None           # Output of digital filtering - spike
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
        self.fs_adc = 0.0           # Sampling rate of the ADC incl. oversampling
        self.fs_dig = 0.0           # Processing rate of the digital part


class invasive_nsp:
    def __init__(self) -> None:
        self.spike_ticks = None     # Spike Ticks
        self.nsp_post = dict()      # Adding some parameters after calculating some neural signal processing methods


class PipelineSignal(invasive_analog, invasive_digital, invasive_nsp):
    def __init__(self) -> None:
        super().__init__()
