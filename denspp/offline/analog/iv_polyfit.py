import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger, Logger
from denspp.offline.plot_helper import scale_auto_value, save_figure
from denspp.offline.analog.dev_noise import SettingsNoise, RecommendedSettingsNoise, ProcessNoise
from denspp.offline.metric.data_numpy import calculate_error_rae


class PolyfitIV(ProcessNoise):
    _logger: Logger
    _settings_noise: SettingsNoise
    _fit_params_v2i: np.ndarray = np.nan    # Voltage input, Current Output
    _fit_params_i2v: np.ndarray = np.nan    # Current input, Voltage Output
    _fit_order: int=3

    def __init__(self, sampling_rate: float, en_noise: bool, settings_noise: SettingsNoise=RecommendedSettingsNoise):
        """Class for extracting the polynom fit parameters of measured IV curve from electrical device
        :param sampling_rate:       Sampling rate [Hz]
        :param en_noise:            Boolean for enabling noise on output signals
        :param settings_noise:      Settings class for handling noise at output
        """
        super().__init__(settings_noise, sampling_rate)
        self._logger = getLogger(__name__)
        self._en_noise = en_noise
        self._settings_noise = settings_noise

    @staticmethod
    def _calc_error(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        """Metric calculation of the Relative Absolute Error (RAE)
        :param y_pred:  Predicted value
        :param y_true:  True value
        :return:        Metric Output
        """
        return calculate_error_rae(y_pred, y_true)

    def _extract_params_for_polynomfit(self, current: np.ndarray, voltage: np.ndarray) -> None:
        """Function for extracting the polynom coefficients for I-V and V-I translation of the electrical device behaviour
        :param current:     Numpy array with the current characteristic of the device [A]
        :param voltage:     Numpy array with the voltage characteristic of the device [V]
        :return:            None
        """
        self._fit_params_v2i = np.polyfit(x=voltage, y=current, deg=self._fit_order)
        self._fit_params_i2v = np.polyfit(x=current, y=voltage, deg=self._fit_order)

    def _test_fit_option(self, voltage_test: np.ndarray, current_test: np.ndarray,
                         methods_compare: list, plot_title: str='',
                         do_plot: bool=True, path2save: str='') -> float:
        """Function for testing and plotting the comparison
        :param voltage_test:            Numpy array with voltage signal
        :param current_test:            Numpy array with current signal
        :param methods_compare:         List with string labels of used method
        :param plot_title:              Title of plot
        :param do_plot:                 Plotting the results of regression and polynom fitting
        :param path2save:               String with path to save the figure
        :return:                        Floating with error value [-1.0 = not available]
        """
        self._logger.debug(f"Make IV comparison: {methods_compare[0]} vs. {methods_compare[1]}")
        i_poly = self.get_current(voltage_test, 0.0)

        error = self._calc_error(i_poly, current_test)
        plot_title_new = f"{plot_title}, 1e3* RAE = {error:.3f}" if plot_title else f"1e3* RAE = {error:.3f}"
        if do_plot:
            self._plot_transfer_function_comparison(
                u_transfer=voltage_test,
                i_dev0=i_poly,
                i_dev1=current_test,
                method_types=methods_compare,
                plot_title=plot_title_new,
                path2save=path2save,
                show_plot=do_plot
            )
        return error

    @staticmethod
    def _plot_transfer_function_comparison(u_transfer: np.ndarray, i_dev0: np.ndarray, i_dev1: np.ndarray,
                                           method_types: list, plot_title: str = '',
                                           path2save: str = '', show_plot: bool = False) -> None:
        """Plotting the transfer function of electrical device for comparison
        :param u_transfer:      Numpy array with voltage from polynom fit (input)
        :param i_dev0:          Numpy array of current response from first method
        :param i_dev1:          Numpy array of current response from second method
        :param method_types:    List with string labels of used methods
        :param plot_title:      String with plot title
        :param path2save:       String with path to save the figure
        :param show_plot:       Showing and blocking the plots [Default: False]
        :return:                None
        """
        scaley, unity = 1.0, '' # scale_auto_value(i_dev1)
        plt.figure()
        plt.tight_layout()

        axs = list()
        axs.append(plt.subplot(2, 1, 1))
        axs.append(plt.subplot(2, 1, 2, sharex=axs[0]))
        axs[0].semilogy(u_transfer, scaley * np.abs(i_dev0), 'r', marker='.', markersize=2,
                        label=f"{method_types[0]} (Current)")
        axs[0].grid()
        axs[0].set_ylabel(r'Current $\log_{10}(I_F)$ / µA')

        axs[1].plot(u_transfer, scaley * i_dev0, 'r', marker='.', markersize=2, label=f"{method_types[0]} (Current)")
        axs[1].grid()
        axs[1].set_ylabel(fr'Current $I_F$ / {unity}A')
        axs[1].set_xlabel(r'Voltage $\Delta U$ / V')

        axs[1].legend()
        axs[0].set_title(plot_title)
        if path2save:
            save_figure(plt, path2save, 'device_iv_charac', ['svg'])
        if show_plot:
            plt.show(block=True)

    def _get_params_for_polynomfit(self, current: np.ndarray, voltage: np.ndarray, do_test: bool=False,
                                   do_plot: bool=False, path2save: str='') -> float:
        """Function to extract the params of electrical device behaviour with polynom fit function
        :param current:     Numpy array with current signal from measurement
        :param voltage:     Numpy array with voltage signal from measurement
        :param do_test:     Performing a test
        :param do_plot:     Plotting the results of regression and polynom fitting
        :param path2save:   String with path to save the figure
        :return:            Floating value with Relative Squared Error
        """
        self._logger.debug(f"Start polynom fitting")
        self._extract_params_for_polynomfit(
            voltage=voltage,
            current=current
        )
        if do_test:
            error = self._test_fit_option(
                voltage_test=voltage,
                current_test=current,
                methods_compare=['Measurement', 'Polynom fitting'],
                plot_title=f"n_p={self._fit_params_i2v.size}",
                do_plot=do_plot,
                path2save=path2save
            )
        else:
            error = -1.0
        return error

    def _find_best_poly_order(self, current: np.ndarray, voltage: np.ndarray, order_start: int, order_stop: int, show_plots: bool=False) -> float:
        """Finding the best polynomial order for fitting
        :param current:         Numpy array with current values
        :param voltage:         Numpy array with voltage values
        :param order_start:     Integer value with starting order number
        :param order_stop:      Integer value with stopping order number
        :param show_plots:      Showing plots of each run
        :return:                None
        """
        self._logger.info("\n=====================================================")
        self._logger.info("Searching the best polynom order with minimal error")
        self._logger.info("=====================================================")
        order_search = [idx for idx in range(order_start, order_stop + 1)]
        error_search = np.zeros_like(order_search, dtype=float)
        for idx, order in enumerate(order_search):
            self.change_fit_settings(order)
            error = self._get_params_for_polynomfit(
                current=current,
                voltage=voltage,
                do_test=True,
                do_plot=show_plots,
                path2save=''
            )
            error_search[idx] = error
            self._logger.info(f"#{idx:02d}: order = {order:02d} --> Error = {error}")

        # --- Finding best order
        xmin = error_search.argmin()
        self._logger.info(f"\nBest solution: Order = {np.array(order_search)[xmin]} with an error of {error_search[xmin]}!")

        # --- Extract params
        self._fit_order = int(np.array(order_search)[xmin])
        return self._get_params_for_polynomfit(
            current=current,
            voltage=voltage,
            do_test=True,
            do_plot=False,
            path2save=''
        )

    def extract_polyfit_params(self, current: np.ndarray, voltage: np.ndarray, show_plots: bool=False,
                               find_best_order: bool=False, order_range: list=(2, 18)) -> float:
        """Extracting the polynom fit parameters and plotting it compared to regression task
        :param current:             Numpy array with current values
        :param voltage:             Numpy array with voltage values
        :param find_best_order:     Find the best poly.-fit order
        :param order_range:         Range with Integer value for search (best polynom order)
        :param show_plots:          Showing plots of each run
        :return:                    Floating value with error
        """
        if not find_best_order:
            return self._get_params_for_polynomfit(
                current=current,
                voltage=voltage,
                do_test=True,
                do_plot=show_plots,
                path2save=''
            )
        else:
            assert len(order_range) == 2, "Parameter: order_range must have a length of two"
            assert order_range[0] <= order_range[1], "Parameter: order_range must have a length of two"
            return self._find_best_poly_order(
                current=current,
                voltage=voltage,
                order_start=order_range[0],
                order_stop=order_range[1],
                show_plots=show_plots,
            )

    def change_fit_settings(self, order: int) -> None:
        """Function for redefining the fit settings
        :param order:    Integer value with starting order
        """
        self._fit_order = order

    def get_voltage(self, current: np.ndarray | float) -> np.ndarray:
        """Getting the voltage response from applied current into device
        :param current:     Numpy array with applied current signal
        :return:            Numpy array with voltage response
        """
        assert self._fit_params_i2v.any() != np.nan, "fit parameters not set - Please call 'get_params_for_polynomfit'"
        v_noise = np.zeros_like(current) if not self._en_noise else self.gen_noise_awgn_dev(current.size, 1e-9)
        return np.poly1d(self._fit_params_i2v)(current) + v_noise

    def get_current(self, voltage_top: np.ndarray | float, voltage_bot: np.ndarray | float) -> np.ndarray:
        """Getting the voltage response from applied current into device
        :param voltage_top:     Numpy array with applied voltage signal on top electrode
        :param voltage_bot:     Numpy array with applied current signal on bottom electrode
        :return:                Numpy array with current response
        """
        assert self._fit_params_v2i.any() != np.nan, "fit parameters not set - Please call 'get_params_for_polynomfit'"
        i_noise = np.zeros_like(voltage_top) if not self._en_noise else self.gen_noise_awgn_dev(voltage_top.size, 1e-9)
        return np.poly1d(self._fit_params_v2i)(voltage_top - voltage_bot) + voltage_bot + i_noise
