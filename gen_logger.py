# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/17 3:17 下午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
import logging
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
import os


def gen_logger(log_file_name, console_printing=False):
    """

    :param log_file_name: str类型，日志文件的名称
    :param console_printing: 布尔类型，是否控制台打印
    :return: 日志记录器
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
        handler = RotatingFileHandler(log_file_name, maxBytes=3 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter(
            '%(process)d %(asctime)s %(levelname)7s %(filename)10s %(lineno)3d | %(message)s ',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        console.setFormatter(formatter)

        logger.addHandler(handler)
        if console_printing:
            logger.addHandler(console)  # 注释掉此行，以避免在控制台打印日志信息

    return logger
