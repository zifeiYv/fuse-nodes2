# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : self_check.py                   #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/14                      #
#                                                                   #
#                     Last Update :                                 #
#                                                                   #
#-------------------------------------------------------------------#
# Desc:                                                             #
#    Before running the program, check the accuracy of              #
#    configuration parameters.                                      #
#                                                                   #
# Classes:                                                          #
#                                                                   #  
# Functions:                                                        #
#                                                                   #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
"""
import pandas as pd

format_ = pd.read_excel('./config_files/format.xlsx', sheet_name=['label', 'rel', 'pro', 'trans'], index_col=0)


class SysNotMatch(Exception):
    pass


def check():
    """用于检查参数有效性的函数"""
    rel: pd.DataFrame
    label: pd.DataFrame
    label, rel, pro, trans = format_.values()  # Extract four data frames
    sys_num = label.shape[1]
    sys_labels = label.columns.values
    print(f"共指定了{sys_num}个系统，其标签分别为：{list(sys_labels)}")

    ent_num = label.shape[0]
    ent_labels = label.index.values
    print(f"共指定了{ent_num}种实体，其标签分别为：{list(ent_labels)}")

    if rel.shape[1] != label.shape[1]:
        raise SysNotMatch("`rel`与`label`中的系统数量不符")
    if not all(rel.columns == label.columns):
        raise SysNotMatch("`rel`,``")
