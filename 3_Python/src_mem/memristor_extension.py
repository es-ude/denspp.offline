import dataclasses
import numpy as np
from package.analog.dev_load import RecommendedSettingsNoise, ElectricalLoad
from scipy.optimize import least_squares


@dataclasses.dataclass
class SettingsMem:
    """Individual data class to configure the memristive device
    Inputs:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
        dev_sel:    Device selection of library with different paramters [0: ext, > 0: internal]
        dev_value:  Dict with params of memristor for "Branch_2" and "Branch_4" with [n [], k [], Is [nA], Rs [kOhm]]
        dev_branch: Selected working area / read branch of device [0 = Branch 2, 1: Branch 4]
        temp:       Temperature [K]
        area:       Area of memristor device [mm^2]
    """
    type:       str
    fs_ana:     float
    noise_en:   bool
    para_en:    bool
    dev_sel:    int
    dev_value:  dict
    dev_branch: int
    temp:       float
    area:       float


RecommendedSettingsDEV = SettingsMem(
    type='Me',
    fs_ana=50e3,
    noise_en=False,
    para_en=False,
    dev_sel=0,
    dev_value={"Sample": {
            "Branch_2": {"n": 22.1, "k": 3.75, "Is": 20.5, "Rs": 401.0},
            "Branch_4": {"n": 8.88, "k": 10.6, "Is": 14.6, "Rs": 107.0}}},
    dev_branch=1,
    temp=300,
    area=0.045
)


class MemristorModel(ElectricalLoad):
    _settings: SettingsMem
    _dev_type: dict
    __print_device = "memristor behaviour"
    _fit_memristor: dict
    _bounds_current: list
    _bounds_voltage: dict
    __poly_fit: np.ndarray

    # --- Definition of samples
    # (n: Nichtidealitätsfaktor, K: ???, Is: Saettigungssperrstrom [nA], Rs: Shunt resistance [kOhm])
    __samples_params = {}
    __samples_params.update({"Sample_001": {
            "Branch_2": {"n": 24.0, "k": 2.75, "Is": 24.5, "Rs": 471.0},
            "Branch_4": {"n": 4.88, "k": 19.6, "Is": 10.6, "Rs": 57.0}}})
    __samples_params.update({"Sample_002": {
            "Branch_2": {"n": 15.8, "k": 0.328, "Is": 28.30, "Rs": 80.4},
            "Branch_4": {"n": 11.4, "k": 2.420, "Is2": 9.36, "Rs": 564.0}}})

    def __init__(self, settings_dev: SettingsMem) -> None:
        """Class for emulating the memristor types from TECHiFAB
        Args:
            settings_dev:   Settings for controlling the device
        """
        super().__init__(settings_dev, RecommendedSettingsNoise)
        self._settings = settings_dev
        self._bounds_current = [-5, -12]
        self._bounds_voltage = {"Branch_2": [-0.6, +5.0], "Branch_4": [-5.0, +0.6]}
        self.__poly_fit = np.zeros((1, ), dtype=float)

        self.__sample_list = list()
        self.__sample_list.append("Sample_Ext")
        for key in self.__samples_params.keys():
            self.__sample_list.append(key)

        self.__branch_list = list()
        first_element = list(self.__samples_params.keys())[0]
        for key in self.__samples_params[first_element].keys():
            self.__branch_list.append(key)

        self._dev_type = self._init_dev()
        self._dev_type.update({f'M0': self.__memristor_tif_v0})
        self._dev_type.update({f'M1': self.__memristor_tif_v1})

    def __extract_iv_curve_from_regression(self, sel_device: int, sel_branch: int,
                                           num_branch_regression=201) -> [np.ndarray, np.ndarray]:
        """Function for getting the I-V curve from regression
        Args:
            sel_device:             Selection of device
            sel_branch:             Selection of branch
            num_branch_regression:  Number of samples for regression
        Returns:
            Two numpy arrays with current and voltage from device
        """
        params = self.__get_params_from_dict(sel_device)
        params_used = params[sel_branch]

        # --- Regression of branch (I-V)
        i_pathn = -np.logspace(self._bounds_current[0], self._bounds_current[1], num_branch_regression, endpoint=True)
        u_pathn = self.__func2reg_memristor(-i_pathn, params_used, np.zeros(i_pathn.shape))
        # --- Concatenate arrays
        i_path = np.concatenate((i_pathn, -np.flipud(i_pathn)[1:]), axis=0)
        u_path = np.concatenate((u_pathn, -np.flipud(u_pathn)[1:]), axis=0)
        # --- Limiting with voltage boundries
        bounds_voltage = self._bounds_voltage[self.__branch_list[sel_branch]]
        x_start = int(np.argwhere(u_path <= bounds_voltage[0])[-1])
        x_stop = int(np.argwhere(u_path >= bounds_voltage[1])[0])

        return i_path[x_start:x_stop], u_path[x_start:x_stop]

    def __get_params_polyfit(self, do_test=False, num_poly_order=11, num_branch_regression=201) -> float:
        """Function to extract the params of memristor behaviour with polyfit function
        Args:
            do_test:                Performing a test
            num_poly_order:         Order for polynominal fit
            num_branch_regression:  Number of samples for fitting regression
        Returns:
            Floating value with MSE
        """
        i_path, u_path = self.__extract_iv_curve_from_regression(
            sel_device=self._settings.dev_sel,
            sel_branch=self._settings.dev_branch,
            num_branch_regression=num_branch_regression
        )
        self.__poly_fit = np.polyfit(x=u_path, y=i_path, deg=num_poly_order)

        # --- Calculating the error-related metric (MSE)
        if do_test:
            bounds_voltage = self._bounds_voltage[self.__branch_list[self._settings.dev_branch]]
            u_poly = np.linspace(bounds_voltage[0], bounds_voltage[1], num_branch_regression, endpoint=True)
            i_poly = np.polyval(self.__poly_fit, u_poly)
            i_test = self.__memristor_tif_v0(u_poly, 0.0)
            di = i_poly - i_test
            mse = np.sum(np.square(di)) / i_poly.size
        else:
            mse = -1.0
        return mse

    def __get_params_from_dict(self, dev_selected: int) -> list:
        """Bringing the params from I-V curve into right format for least_squares()
        Args:
            dev_selected:   Selected sample or device
        Returns:
            List with parameters of memristor device
        """
        param_in = self.__samples_params if dev_selected != 0 else self._settings.dev_value
        param_in = param_in[self.__sample_list[dev_selected]]

        param_out = list()
        for branch, param_branch in param_in.items():
            param = list()
            param.append(param_branch['n'])
            param.append(param_branch['k'])
            param.append(param_branch['Is'])
            param.append(param_branch['Rs'])
            param_out.append(param)
        return param_out

    def __func2reg_memristor(self, i_path: np.ndarray, params: list, xd: np.ndarray) -> np.ndarray:
        """Function for do least_squared regression
        Args:
            i_path: Input current sample (--> output value)
            params: Parameter for optimization
            xd:     Input difference voltage sample
        Returns:
            Numpy value with corresponding difference voltage (goes to zero with further optimization)
        """
        n_eff = params[0] + params[0] * params[1] ** self.temperature_voltage * np.log(i_path / (1e-9 * params[2]) + 1)
        v1 = n_eff * self.temperature_voltage * np.log(i_path / (1e-9 * params[2]) + 1)
        v3 = 1e3 * params[3] * i_path
        return xd - v1 - v3

    def plot_fit_curve(self, num_samples_regression=201) -> None:
        """Plotting the output of the polynominal fit function
        Args:
            num_samples_regression:  Num of samples for regression
        Returns:
            None
        """
        # --- Getting curve from regression
        i_path, u_path = self.__extract_iv_curve_from_regression(
            sel_device=self._settings.dev_sel,
            sel_branch=self._settings.dev_branch,
            num_branch_regression=num_samples_regression
        )
        # --- Test
        mse = self.__get_params_polyfit(True)
        bounds_v = self._bounds_voltage[self.__branch_list[self._settings.dev_branch]]
        u_poly = np.linspace(bounds_v[0], bounds_v[1], i_path.size, endpoint=True)
        i_poly = self.get_current(u_poly, 0.0)

        # --- Plotting
        plt.figure()
        axs = list()
        axs.append(plt.subplot(2, 1, 1))
        axs.append(plt.subplot(2, 1, 2, sharex=axs[0]))
        axs[0].semilogy(u_path, 1e6 * abs(i_path), 'k', marker='.', markersize=2, label='Regression')
        axs[0].semilogy(u_poly, 1e6 * abs(i_poly), 'r', marker='.', markersize=2, label='Poly. fit')
        axs[0].grid()
        axs[0].set_ylabel(r'Current $log10(I_F)$ / µA')

        axs[1].plot(u_path, 1e6 * i_path, 'k', marker='.', markersize=2, label='Regression')
        axs[1].plot(u_poly, 1e6 * i_poly, 'r', marker='.', markersize=2, label='Poly. fit')
        axs[1].grid()
        axs[1].set_ylabel(r'Current $I_F$ / µA')
        axs[1].legend()
        axs[0].set_title(f"{self.__branch_list[self._settings.dev_branch]} "
                         f"with sqrt(MSE) = {1e9 * np.sqrt(mse):.4f} nA")

        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')
        plt.tight_layout()
        plt.show()

    def __memristor_tif_v0(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a characterized memristor TiF device
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        params = self.__get_params_from_dict(self._settings.dev_sel)
        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)

        # --- Start Conditions
        param_used = params[self._settings.dev_branch]
        bounds = [10 ** self._bounds_current[1], 10 ** self._bounds_current[1]]
        y_initial = 1e-6

        # --- Run optimization
        iout = list()
        for idx, u_sample in enumerate(du):
            sign_pos = u_sample >= 0.0
            y_start = y_initial if idx == 0 else abs(iout[-1])
            result = least_squares(self.__func2reg_memristor, y_start, jac='3-point', bounds=(bounds[0], bounds[1]),
                                   args=(param_used, abs(u_sample)))
            iout.append(result.x[0] if sign_pos else -result.x[0])
        return np.array(iout, dtype=float)

    def __memristor_tif_v1(self, u_inp: np.ndarray | float, u_inn: np.ndarray | float) -> np.ndarray:
        """Performing the behaviour of a characterized memristor TiF device
        Args:
            u_inp:   Positive input voltage [V]
            u_inn:   Negative input voltage [V]
        Returns:
            Corresponding current signal
        """
        du = u_inp - u_inn
        if isinstance(du, float):
            du = list()
            du.append(u_inp - u_inn)
        if self.__poly_fit.size == 1:
            self.__get_params_polyfit()
        return np.polyval(self.__poly_fit, du)


# ----------------------------- EXTERNAL CODE FOR LOCAL TEST ------------------------------------------
def __plot_test_results(time: np.ndarray, u_in: np.ndarray, i_in: np.ndarray,
                        mode_current_input: bool, do_ylog=False) -> None:
    """Only for testing"""
    scale_i = 1e6
    scale_u = 1

    signalx = scale_i * i_in if mode_current_input else scale_u * u_in
    signaly = scale_u * u_in if mode_current_input else scale_i * i_in
    label_axisx = 'Voltage U_x [V]' if mode_current_input else 'Current I_x [µA]'
    label_axisy = 'Current I_x [µA]' if mode_current_input else 'Voltage U_x [V]'
    label_legx = 'i_in' if mode_current_input else 'u_in'
    label_legy = 'u_out' if mode_current_input else 'i_out'

    # --- Plotting: Transient signals
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(time[0], time[-1])
    twin1 = axs[0].twinx()
    a = axs[0].plot(time, signalx, 'k', label=label_legx)
    axs[0].set_ylabel(label_axisy)
    axs[0].set_xlabel('Time t [s]')
    b = twin1.plot(time, signaly, 'r', label=label_legy)
    twin1.set_ylabel(label_axisx)
    axs[0].grid()

    # Generate common legend
    lns = a + b
    labs = [l0.get_label() for l0 in lns]
    axs[0].legend(lns, labs, loc=0)

    # --- Plotting: I-U curve
    if mode_current_input:
        if do_ylog:
            axs[1].semilogy(signaly, abs(signalx), 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signaly, signalx, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisx)
        axs[1].set_ylabel(label_axisy)
    else:
        if do_ylog:
            axs[1].semilogy(signalx, abs(signaly), 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signalx, signaly, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisy)
        axs[1].set_ylabel(label_axisx)
    axs[1].grid()

    plt.tight_layout()


def __generate_signal(t_end: float, fs: float, upp: list, fsig: list, uoff=0.0) -> [np.ndarray, np.ndarray]:
    """Generating a signal for testing
    Args:
        t_end:      End of simulation
        fs:         Sampling rate
        upp:        List with amplitude values
        fsig:       List with corresponding frequency
        uoff:       Offset voltage
    """
    t0 = np.linspace(0, t_end, num=int(t_end * fs), endpoint=True)
    uinp = np.zeros(t0.shape) + uoff
    for idx, peak_val in enumerate(upp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * fsig[idx])
    return t0, uinp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mem_set = SettingsMem(
        type='M0',
        fs_ana=2000e3,
        noise_en=False,
        para_en=False,
        dev_sel=1,
        dev_value={},
        dev_branch=1,
        temp=300,
        area=0.045
    )

    # --- Declation of input
    # t0, uinp = __generate_signal(0.5e-3, mem_set.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 2.5)
    t0, uinp = __generate_signal(0.5e-3, mem_set.fs_ana, [2.5], [10e3], 3.0)
    uinn = 0.0

    # --- Model declaration
    mem_dev = MemristorModel(mem_set)
    mem_dev.plot_fit_curve()
    iout = mem_dev.get_current(uinp, uinn)

    # --- Plotting: Current response
    plt.close('all')
    __plot_test_results(t0, uinp - uinn, iout, False, True)
    plt.show()
