import numpy as np
from src_mem.pipeline_mem import PipelineSignal
from src_mem.pipeline_v1 import Pipeline


if __name__ == "__main__":
    fs_ana = 50e3

    # --- Definition of DUT
    dut = Pipeline(fs_ana)

    # --- Declaration of input
    t_end = 10e-3
    t0 = np.linspace(0, t_end, num=int(t_end * fs_ana), endpoint=True)
    u_off = 0.0
    u_pp = [0.25, 0.3, 0.1]
    f0 = [1e3, 1.8e3, 2.8e3]
    uinp = np.zeros(t0.shape) + u_off
    for idx, peak_val in enumerate(u_pp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * f0[idx])
    uinn = 0.0

    # --- Run and plot
    dut.run(uinp, 0.0)
    data = dut.signals

    print("Done")
