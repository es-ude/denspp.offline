import logging
from os.path import join


def define_logger_testing(save_file: bool=True):
    """Function for preparing the logger configuration in testing routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.DEBUG,
        filename='run_test_report.log' if save_file else None,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


def define_logger_runtime(save_file: bool=True):
    """Function for preparing the logger configuration in runtime routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.INFO,
        filename=join('runs', 'runtime_report_normal.log') if save_file else None,
        filemode='w',
        format='%(asctime)s: %(message)s'
    )


def define_logger_runtime_debug(save_file: bool=True):
    """Function for preparing the logger configuration in runtime debugging routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.DEBUG,
        filename=join('runs', 'runtime_report_debug.log') if save_file else None,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


def define_logger_training(path_save: str, save_file: bool=True):
    """Function for preparing the logger configuration in ML training routines
    :param path_save:   Path where the output should be saved
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.INFO,
        filename=join(path_save, 'run_training_report.log') if save_file else None,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )