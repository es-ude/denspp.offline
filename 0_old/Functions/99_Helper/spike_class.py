import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class spike_dataclass():
    def __init__(self, data: dict):
        self.raw = data['raw_data']
        self.sampling_rate = data['sampling_rate']
        if 'sampling_rate' in data.keys():
            self.times = data['spike_times']
        if 'sampling_rate' in data.keys():
            self.cluster = data['spike_cluster']

    def align_spike_frames(self):
        aligned_frames = list()
        for x in self.times:
            aligned_frames.append(self.get_spikeframe(x))
        self.aligned_spikes = list(aligned_frames)

    def get_spikeframe(self, spike_time: int) -> np.ndarray:
        #be careful. Not aligned. Does not look sophisticating
        window_width = 60
        spike_frame = self.raw[range(spike_time, spike_time + window_width)]
        ###align
        aligned_frame = self.alignment(spike_frame)
        return aligned_frame

    def alignment(self, spike_frame: np.ndarray, mode: str = "max"):
        if mode == "max":
            align_start = 18
            align_range = 7
            align_window_width = range(align_start, align_start + align_range)
            align_frame = spike_frame[align_window_width]
            if max(align_frame) != align_frame[0] and max(align_frame) != align_frame[-1]:
                align = np.where(align_frame == max(align_frame))[0][0]
            else:
                align = np.where(align_frame == min(align_frame))[0][0]
            align_min = align_start + align - 15
            align_max = align_start + align + 32
            aligned_frame = spike_frame[align_min:align_max]
            return aligned_frame
        elif mode == "max_diff":
            pass