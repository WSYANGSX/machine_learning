import sys
import logging


def set_logger(name: str, verbose: bool = True):
    logger = logging.getLogger(name)

    level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


LOGGER = set_logger("Machine learning")
