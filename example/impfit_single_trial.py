import numpy as np
from denspp.offline import YamlConfigHandler
from src_impfit.imp_fitting.impfitter_handler import RecommendedSettingsImpFit, SettingsImpFit, ImpFitHandler


if __name__ == "__main__":
    yaml_config = YamlConfigHandler(RecommendedSettingsImpFit, yaml_name="Config_ImpFit_Test")
    settings_impfit = yaml_config.get_class(SettingsImpFit)

    path2imp = settings_impfit.path2fits
    path2ngsolve = f'{path2imp}/impedance_expected_ngsolve.csv'
    path2test0 = f'{path2imp}/tek0000ALL_MATLAB_new_fit.csv'
    path2test1 = f'{path2imp}/tek0000ALL_MATLAB_impedance.csv'

    imp_handler = ImpFitHandler()
    imp_handler.load_fitmodel(settings_impfit.model)
    imp_handler.load_params_default(path2ngsolve, {'ct_R': 8.33e6})

    fit2freq = np.logspace(0, 6, 61, endpoint=True)
    z_prd = imp_handler.do_impedance_fit_from_params(imp_handler.get_params_default(), fit2freq)
    z_fit0 = imp_handler.do_impedance_fit_from_params_csv(path2test0, fit2freq)
    z_fit1 = imp_handler.do_impedance_fit_from_predicted_csv(path2test1, fit2freq)

    # --- Step #1: Plotting impedance
    imp_handler.plot_impedance_results(imp_stim=z_fit0, imp_mod=z_prd, plot_name='comparison', show_plot=True)
