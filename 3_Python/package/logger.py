"""Module providing logging support."""

import logging
from logdecorator import log_on_end, log_on_error, log_on_start


@log_on_start(logging.DEBUG, "Try updating logging configuration")
@log_on_error(
    logging.ERROR,
    "Error on updaing logging configuration: {e!r}",
    on_exceptions=[Exception],
    reraise=True,
)
@log_on_end(
    logging.INFO,
    "Set log level to " + logging.getLevelName(logging.root.level),
)
@log_on_end(logging.INFO, "Set log file to {log_file}")
def setup_logging(level: int = 0, log_file: str | None = None) -> None:
    """Set logger configuration.

    Args:
        level (int): log level to use (default=0)
                     (0=ERROR; 1=WARNING; 2=INFO ; 3=DEBUG)
        log_file (str): set file to log, if None log to CLI (default=None)
    """
    if level == 0:
        log_level = logging.ERROR
    elif level == 1:
        log_level = logging.WARNING
    elif level == 2:
        log_level = logging.INFO
    elif level == 3:
        log_level = logging.DEBUG
    else:
        log_level = None

    if log_file is not None:
        logging.basicConfig(filename=log_file, level=log_level)
    else:
        logging.basicConfig(level=log_level)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_logging(3, None)
