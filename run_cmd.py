# -*- coding: utf-8 -*-
"""
=============
执行融合的主程序
=============
"""
import argparse
from self_test import check
from subgraphs import delete_old
from configparser import ConfigParser
from multiprocessing import Pool

parser = argparse.ArgumentParser()

parser.add_argument('--new-label', type=str, required=True, help='生成的融合图的标签')

args = parser.parse_args()
label = args.new_label

if __name__ == '__main__':
    print("检查配置文件...")
    processes = check()
    print("完成\n")
    print("删除旧结果...")
    delete_old(label)
    print("删除完成\n")

    cfg = ConfigParser()

    with open('./config_files/application.cfg', encoding='utf-8') as f:
        cfg.read_file(f)

    print("开始融合")
    from utils import fuse_root_nodes, fuse_and_create
    root_results = fuse_root_nodes()

    if processes == 1:
        print("单进程融合")
        for i in range(len(root_results)):
            fuse_and_create((label, root_results[i], i, len(root_results)))
    else:
        print("多进程融合")
        p = Pool(processes=processes)
        for i in range(len(root_results)):
            p.apply_async(fuse_and_create, args=((label, root_results[i], i, len(root_results)),))
        p.close()
        p.join()
