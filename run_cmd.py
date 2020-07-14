# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/28 10:55 上午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
import pickle
from self_test import check
from subgraphs import delete_old
import os


if __name__ == '__main__':
    print("Checking config files...")
    check()
    print("Everything goes well\n")
    print("Deleting old results...")
    delete_old('merge')
    print("Finish delete\n")

    with open('./config_files/application.cfg') as f:
        cfg.read_file(f)
    processes = cfg.getint('distributed', 'processes')
    if not processes:
        processes = os.cpu_count()
    print("Fusing started")
    from utils import fuse_root_nodes
    root_results = fuse_root_nodes(multi, processes)
    if root_results is None:
        print("Foot nodes have not fuse results")
    if multi:
        inputs = [[label, root_results[i], i, len(root_results)] for i in range(len(root_results))]
        with open('./inputs.pkl', 'wb') as f:
            pickle.dump(inputs, f)
        os.popen('nohup python multiprocess.py > multi.out 2>&1 &')  # 标准错误重定向至标准输出，不适合windows操作系统
    else:
        from utils import fuse_and_create
        for i in range(len(root_results)):
            fuse_and_create((label, root_results[i], i, len(root_results)))
