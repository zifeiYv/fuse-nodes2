# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/28 10:55 上午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
import argparse
import pickle
from self_test import check
from subgraphs import delete_old
from configparser import ConfigParser
import os


parser = argparse.ArgumentParser()

parser.add_argument('--new-label', type=str, required=True, help='生成的融合图的标签')
parser.add_argument('--multi', type=int, default=0, choices=[0, 1], help='是否采用多进程加速运算，默认不采用')

args = parser.parse_args()
label = args.new_label
multi = args.multi
# label = 'yh'
# multi = 0


if __name__ == '__main__':
    print("Checking config files...")
    check()
    print("Everything goes well\n")
    print("Deleting old results...")
    delete_old(label)
    print("Finish delete\n")

    cfg = ConfigParser()

    with open('./config_files/application.cfg') as f:
        cfg.read_file(f)
    processes = cfg.getint('distributed', 'processes')
    if not processes:
        processes = os.cpu_count()
    print("****************Fusing root data firstly****************")
    from utils import fuse_root_nodes
    root_results = fuse_root_nodes(multi, processes)
    if multi:
        inputs = [[label, root_results[i], i, len(root_results)] for i in range(len(root_results))]
        with open('./inputs.pkl', 'wb') as f:
            pickle.dump(inputs, f)
        os.popen('nohup python multiprocess.py > multi.out 2>&1 &')  # 标准错误重定向至标准输出，不适合windows操作系统
    else:
        from utils import fuse_and_create
        for i in range(len(root_results)):
            fuse_and_create((label, root_results[i], i, len(root_results)))
