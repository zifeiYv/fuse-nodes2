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
#                     Last Update : 2020/07/14                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
import pandas as pd
from numpy import nan
from py2neo import Graph
from pymysql import connect
from configparser import ConfigParser

format_ = pd.read_excel('./config_files/format.xlsx', sheet_name=['label', 'rel', 'pro', 'trans'],
                        index_col=0)

cfg = ConfigParser()
with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
mysql = cfg.get('mysql', 'mysql')


class CheckError(Exception):
    pass


def check():
    """用于检查参数有效性的函数"""
    label, rel, pro, trans = format_.values()  # Extract four data frames
    label.replace(r'^\s+$', nan, regex=True, inplace=True)
    rel.replace(r'^\s+$', nan, regex=True, inplace=True)
    pro.replace(r'^\s+$', nan, regex=True, inplace=True)
    trans.replace(r'^\s+$', nan, regex=True, inplace=True)

    sys_num = label.shape[1]
    sys_labels = label.columns.values
    print(f"共指定了{sys_num}个系统，其标签分别为：{list(sys_labels)}")
    if sys_num < 2:
        raise CheckError("至少需要指定两个系统才能进行融合")

    ent_num = label.shape[0]
    print(f"共指定了{ent_num}种实体")

    for df in ['rel', 'pro', 'trans']:
        df_ = eval(df)
        if df_.shape[1] != label.shape[1]:
            raise CheckError(f"`{df}`与`label`中的系统数量不符")
        # noinspection PyTypeChecker
        if not all(df_.columns == label.columns):
            raise CheckError(f"`{df}`与`label`中的系统名称不符")

    for s in sys_labels:
        if label[s].value_counts().sum() != rel[s].value_counts()[0] + 1:
            raise CheckError(f"系统`{s}`的实体类别数量与关系数量不匹配")
        if label[s].value_counts().sum() != pro[s].value_counts().sum():
            raise CheckError(f"系统`{s}`可用于融合计算的属性数量与实体类别数量不匹配")
        if label[s].value_counts().sum() != trans[s].value_counts().sum():
            raise CheckError(f"系统`{s}`融合后需迁移的属性与实体类别数量不匹配")

    for i in range(pro.shape[0]):
        len_ = None
        for j in range(pro.shape[1]):
            if isinstance(pro.iloc[i, j], str):
                if len_:
                    if len(pro.iloc[i, j].split(',')) != len_:
                        raise CheckError(f"{i+1}级实体可用于融合计算的属性数量不一致")
                    else:
                        len_ = len(pro.iloc[i, j].split(','))

    try:
        graph = Graph(neo4j_url, auth=auth)
        graph.run("Return 'OK'")
    except Exception as e:
        raise CheckError(e)

    try:
        connect(**eval(mysql))
    except Exception as e:
        raise CheckError(e)
