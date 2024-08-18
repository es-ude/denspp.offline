import numpy as np
import matplotlib.pyplot as plt


def plot_iv_curve(u_in: np.ndarray, i_out: np.ndarray, do_log=False, path2save='') -> None:
    """"""
    plt.figure()
    if not do_log:
        plt.plot(u_in, i_out)
    else:
        plt.semilogy(u_in, np.abs(i_out))
    plt.xlabel(r'Voltage $U$ / V')
    plt.ylabel(r'Current $I$ / A')

    plt.tight_layout()
    plt.grid()
    if path2save:
        plt.savefig('test.svg', format='svg')
    plt.show(block=True)


def plot_transient(time: np.ndarray, v_in: np.ndarray, v_out: np.ndarray, i_in: np.ndarray, path2save='') -> None:
    """"""
    fig, ax1 = plt.subplots()
    ax1.plot(time, v_in, label='inp')
    ax1.plot(time, v_out, label='out')
    ax1.set_ylabel(r"Voltage $U$ / V")
    ax1.set_xlabel(r'Time $t$ / s')

    ax2 = ax1.twinx()
    ax2.plot(time, i_in, 'r--', label='current')
    ax2.set_ylabel(r"Current $I$ / A")

    plt.legend()
    plt.tight_layout()
    plt.grid()
    if path2save:
        plt.savefig('test.svg', format='svg')
    plt.show(block=True)


def plot_bodeplot(freq: np.ndarray, v_in: np.ndarray, v_out: np.ndarray, path2save='') -> None:
    """"""
    transfer_function = v_out / v_in
    fig, ax1 = plt.subplots()
    ax1.semilogx(freq, 20 * np.log10(np.abs(transfer_function)), 'k')
    ax1.set_ylabel(r"Gain $v_U$ / dB")
    ax1.set_xlabel(r'Frequency $f$ / dB')

    ax2 = ax1.twinx()
    ax2.semilogx(freq, np.angle(transfer_function, deg=True), 'r')
    ax2.set_ylabel(r"Phase $\alpha$ / °")

    plt.tight_layout()
    plt.grid()
    if path2save:
        plt.savefig('test.svg', format='svg')
    plt.show(block=True)