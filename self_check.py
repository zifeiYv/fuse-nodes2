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
#                     Last Update : 2020/08/06                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
import pandas as pd
from numpy import nan
from py2neo import Graph
from pymysql import connect
from configparser import ConfigParser


cfg = ConfigParser()
with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
mysql_cfg = cfg.get('mysql', 'mysql_cfg')
mysql_res = cfg.get('mysql', 'mysql_res')


class CheckError(Exception):
    pass


def check(task_id):
    """用于检查参数有效性的函数"""
    # 数据库连接是否正常
    for i in (mysql_res, mysql_cfg):
        try:
            connect(**eval(i))
        except Exception as e:
            raise CheckError(e)

    label, pro, trans, rel = get_paras(task_id)  # 从任务id，获取相关参数并处理成DataFrame的格式

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
        if label[s].value_counts().sum() != rel[s].value_counts().sum() + 1:
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


def get_paras(task_id):
    """从关系型数据库获取参数，并处理成DataFrame的形式返回给``check()``进行调用"""
    try:
        conn = connect(**eval(mysql_cfg))
    except Exception as e:
        raise CheckError(e)
    with conn.cursor() as cr:
        cr.execute(f"select id, system_label, entity_label, pros_for_fuse, pros_for_transfer, "
                   f"entity_level from fuse_config_table1 where task_id='{task_id}'")
        info = cr.fetchall()
        cr.execute(f"select from_id, to_id, rel_label from fuse_config_table2 "
                   f"where task_id='{task_id}'")
        info2 = cr.fetchall()
    num_rows = max([i[-1] for i in info])
    columns = list(set([i[1] for i in info]))
    label = pd.DataFrame(data=nan, index=range(num_rows), columns=columns)
    pro = pd.DataFrame(data=nan, index=range(num_rows), columns=columns)
    trans = pd.DataFrame(data=nan, index=range(num_rows), columns=columns)
    rel = pd.DataFrame(data=nan, index=range(num_rows-1), columns=columns)
    for c in columns:
        for t in info:
            if t[1] != c:
                continue
            else:
                label.loc[t[-1]-1, c] = t[2]
                pro.loc[t[-1]-1, c] = t[3]
                trans.loc[t[-1]-1, c] = t[4]
                for t2 in info2:
                    if t2[0] == t[0]:
                        rel.loc[t[-1]-1, c] = t2[2]

    return label, pro, trans, rel
