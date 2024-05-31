class PipelineSignal:
    def __init__(self) -> None:
        self.fs_ana = 0.0       # "Sampling rate"
        self.fs_adc = 0.0       # Sampling rate of the ADC incl. oversampling
        self.fs_dig = 0.0       # Processing rate of the digital part

        self.u_inp = None       # Positive input voltage
        self.u_inn = None       # Negative input voltage
        self.u_pre = None       # Output of pre-amp
        self.u_dly = None       # Delayed signal
        self.u_spk = None       # Output of analogue filtering - spike acitivity
        self.u_cmp = None

        self.u_mem_top = None
        self.u_mem_bot = None
        self.i_tra = None
        self.u_tra = None
        self.u_int = None
        self.x_feat = None      # ADC output
