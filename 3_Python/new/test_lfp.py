import numpy as np
import matplotlib.pyplot as plt


R0 = 100e3
C0 = 220e-12
wf = np.linspace(0.001, 0.5, 128, endpoint=True)
wg = np.linspace(0.001, 1.0, 256, endpoint=True)

fg = 1 / (2 * np.pi * np.sqrt(wf - wf * wf) * R0 * C0)
gain = - wg / (1 - wg + 100)

plt.figure()
plt.semilogy(wf, 1e-3 * fg, linestyle='none', marker='.', label="LFP")

plt.xlabel("Potentiometer Position")
plt.ylabel(r"f_g [kHz]")
plt.grid()
plt.tight_layout()

plt.figure()
plt.semilogy(wg, np.abs(gain), linestyle='none', marker='.', label="LFP")

plt.xlabel("Potentiometer Position")
plt.ylabel(r"A_v [V/V]")
plt.grid()
plt.tight_layout()

plt.show()
