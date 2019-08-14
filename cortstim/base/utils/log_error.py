# Logs and errors

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from cortstim.base.config.config import OutputConfig


def initialize_logger(name, target_folder=OutputConfig().FOLDER_LOGS):
    """
    create logger for a given module
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not (os.path.isdir(target_folder)):
        os.makedirs(target_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.CRITICAL)

    fh = TimedRotatingFileHandler(
        os.path.join(
            target_folder,
            'logs.log'),
        when="d",
        interval=1,
        backupCount=2)
    fh.setFormatter(formatter)
    fh.setLevel(logging.CRITICAL)

    # Log errors separately, to have them easy to inspect
    fhe = TimedRotatingFileHandler(
        os.path.join(
            target_folder,
            'log_errors.log'),
        when="d",
        interval=1,
        backupCount=2)
    fhe.setFormatter(formatter)
    fhe.setLevel(logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fhe)

    return logger


def raise_value_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nValueError: " + msg + "\n")
    raise ValueError(msg)


def raise_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nError: " + msg + "\n")
    raise Exception(msg)


def raise_import_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nImportError: " + msg + "\n")
    raise ImportError(msg)


def raise_not_implemented_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nNotImplementedError: " + msg + "\n")
    raise NotImplementedError(msg)


def warning(msg, logger=None):
    if logger is not None:
        logger.warning("\n" + msg + "\n")
