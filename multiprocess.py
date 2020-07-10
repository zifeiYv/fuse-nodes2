# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/28 1:24 下午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
from multiprocessing import Pool
import pickle
from configparser import ConfigParser
import os
from utils import fuse_and_create

cfg = ConfigParser()

with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
processes = cfg.getint('distributed', 'processes')
if processes:
    print(f"Using multiple processes to speed fuse, number of process is {processes}")
else:
    processes = os.cpu_count()
    print(f"Using multiple processes to speed fuse, do not specify number of process, default:{processes}")


if __name__ == '__main__':
    with open('./inputs.pkl', 'rb') as f:
        inputs = pickle.load(f)
    pool = Pool(processes)
    pool.map(fuse_and_create, inputs)
    pool.close()
    pool.join()
