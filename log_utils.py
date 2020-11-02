# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
import os


def gen_logger(log_file_name, console_printing=False):
    """Generate a `logging.getLogger` object to log something.

    Args:
        log_file_name(str): Name of log file
        console_printing(bool): Whether to print log on console

    Returns:
        A `logging.getLogger` object.
    """
    cwd = os.getcwd()
    path = os.path.join(cwd, 'logs')
    if not os.path.exists(path):
        os.makedirs(path)
    log_file_name = path + '/' + log_file_name

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # 避免重复添加handler
        console = StreamHandler()
        handler = RotatingFileHandler(log_file_name, maxBytes=3 * 1024 * 1024,
                                      backupCount=5)
        formatter = logging.Formatter(
            '%(process)d %(asctime)s %(levelname)7s %(filename)10s %(lineno)3d |'
            ' %(message)s ',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        console.setFormatter(formatter)

        logger.addHandler(handler)
        if console_printing:
            logger.addHandler(console)
    return logger
