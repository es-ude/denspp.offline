import logging
from os.path import join


def define_logger_testing():
    return logging.basicConfig(
        level=logging.DEBUG,
        filename='run_test_report.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


def define_logger_runtime(save_file: bool=True):
    return logging.basicConfig(
        level=logging.INFO,
        filename=join('runs', 'runtime_report_normal.log') if save_file else None,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


def define_logger_runtime_debug():
    return logging.basicConfig(
        level=logging.DEBUG,
        filename=join('runs', 'runtime_report_debug.log'),
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


def define_logger_training(path_save: str):
    return logging.basicConfig(
        level=logging.INFO,
        filename=join(path_save, 'run_test_report.log'),
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )