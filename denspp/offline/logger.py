import logging
from pathlib import Path

from denspp.offline import get_path_to_project


def define_logger_testing(
    level: int = logging.DEBUG, path2save: Path = get_path_to_project(), save_file: bool = False
):
    """Function for preparing the logger configuration in testing routines
    :param level:       Logging level
    :param path2save:   Path for saving the outputs
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    path2log = path2save.absolute() / "report_testing.log"
    return logging.basicConfig(
        level=level,
        filename=path2log.as_posix() if save_file else None,
        filemode="w",
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
    )


def define_logger_runtime(
    level: int = logging.INFO, path2save: Path = get_path_to_project(), save_file: bool = False
):
    """Function for preparing the logger configuration in runtime routines
    :param level:       Logging level
    :param path2save:   Path for saving the outputs
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    path2log = path2save.absolute() / "report_runtime.log"
    return logging.basicConfig(
        level=level,
        filename=path2log.as_posix() if save_file else None,
        filemode="w",
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
    )
