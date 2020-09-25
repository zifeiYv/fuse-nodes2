# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/28 10:55 上午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
import argparse
from self_test import check
from subgraphs import delete_old
from configparser import ConfigParser


parser = argparse.ArgumentParser()

parser.add_argument('--new-label', type=str, required=True, help='生成的融合图的标签')

args = parser.parse_args()
label = args.new_label

if __name__ == '__main__':
    print("检查配置文件...")
    check()
    print("完成\n")
    print("删除旧结果...")
    delete_old(label)
    print("删除完成\n")

    cfg = ConfigParser()

    with open('./config_files/application.cfg', encoding='utf-8') as f:
        cfg.read_file(f)

    print("开始融合")
    from utils import fuse_root_nodes
    root_results = fuse_root_nodes()
    from utils import fuse_and_create
    for i in range(len(root_results)):
        fuse_and_create((label, root_results[i], i, len(root_results)))
