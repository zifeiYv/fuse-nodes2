# -*- coding: utf-8 -*-
"""
File Name  : fuse
Author     : Jiawei Sun
Email      : j.w.sun1992@gmail.com
Start Date : 2020/07/14
Describe   :
    执行融合任务的主要函数和方法
"""
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
from configparser import ConfigParser
from utils import Nodes, sort_sys, Computation
from self_check import get_paras
from progressbar import ProgressBar
from pymysql import connect
from time import strftime
from uuid import uuid1
import json
import requests

LABEL, PRO, TRANS = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
fused_label = ''
BASE_SYS_ORDER = {}
cfg = ConfigParser()
with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
threshold = float(cfg.get('threshold', 'threshold'))
mysql_res = cfg.get('mysql', 'mysql_res')
mysql_cfg = cfg.get('mysql', 'mysql_cfg')
err_url = cfg.get('error_handler', 'url')


def main_fuse(task_id):
    global LABEL, PRO, TRANS, BASE_SYS_ORDER, fused_label
    LABEL, PRO, TRANS, fused_label = get_paras(task_id)

    BASE_SYS_ORDER = sort_sys(LABEL)
    bar = ProgressBar(fused_label)
    print("开始融合")
    print("删除旧的融合结果...")
    delete_old(fused_label)
    print("删除完成")
    print("正在融合根节点")
    start_time = strftime("%Y-%m-%d %H:%M:%S")
    mapping, next_batch_no = get_save_mapping(task_id)
    counter_only = pd.DataFrame(data=0, columns=LABEL.columns, index=LABEL.index)
    counter_all = pd.DataFrame(data=0, columns=LABEL.columns, index=LABEL.index)
    try:
        root_res_df = fuse_root_nodes()
    except Exception as e:
        err_data = json.dumps({
            "task_id": task_id,
            "state": "xxx",
            "msg": "根节点融合出错",
            "value": e
        })
        requests.post(err_url, data=err_data)
        return
    else:
        if root_res_df.empty:
            print("根节点融合后无结果，无法继续执行")
            err_data = json.dumps({
                "task_id": task_id,
                "state": "xxx",
                "msg": "根节点融合后无结果",
                "value": ""
            })
            requests.post(err_url, data=err_data)
            return
        else:
            print("根节点融合完成，开始融合子图")
            try:
                base_ent_lab = LABEL[BASE_SYS_ORDER[0]].iloc[0]
                for i in range(len(root_res_df)):
                    bar.set((i + 1)/len(root_res_df))
                    node = Nodes(base_ent_lab, root_res_df.iloc[i].to_list())
                    fuse_other_nodes(1, node, BASE_SYS_ORDER)  # 执行之后，node包含了创建一个子图所需要的完整信息
                    a, b = caching(counter_only, counter_all, node)
                    counter_only.append(a)
                    counter_all.append(b)
                    create_node_and_rel(node)
                save_res_to_mysql(counter_only, counter_all, mapping, next_batch_no, task_id, start_time)
                print("创建新图完成")
            except Exception as e:
                err_data = json.dumps({
                    "task_id": task_id,
                    "state": "xxx",
                    "msg": "执行过程中出错",
                    "value": e
                })
                requests.post(err_url, data=err_data)
                return


def fuse_root_nodes(label: pd.DataFrame = None, base_order=0, not_extract=None):
    """对根节点进行融合。

    根节点指的是各个系统中的第一个实体类，为了加快融合速度，应该增加额外的信息以
    帮助程序缩小检查范围，如行政区域划分等。

    融合过程需要递归进行，即逐层获取基准系统与目标系统进行相似度计算，获取融合结果，
    最后再将融合结果统一整理。

    假设有5个系统需要融合：
              | sys1 | sys2 | sys3 | sys4 | sys5 |
        ------|------|------|------|------|------|
        level1| Ent  | Ent  | Ent  | Ent  | Ent  |
        ------|------|------|------|------|------|
        ...
    根节点的融合只需要考察1级实体的标签即可。

    第一步：
        假设通过判断后，sys1为基准系统，其他为目标系统，则需要分别计算sys1与其他系统
        之间的节点的相似度。由于sys1是基准系统，那么其他系统中判定为与sys1为相同实体
        的节点之间自动判定为相同实体。
    第二步：
        在基准系统与其他系统融合的结果中，只保证全部包含基准系统中的节点，不必全部包含
        其他系统的节点。如sys1与sys2的结果，若sys1中有3个节点，sys2中有四个节点，那
        么其融合结果可以是：[[0, 1], [1, None], [2, 2]]
    第三步：
        将两两系统的融合结果进行组合与拼接，其结果格式如下：
        [[0, 1, None, None, 1],
         [1, None, 1, 2, 0],
         [2, 2, 2, 0, 2]]
        注意，其中存储的数字为从Neo4j中读取的节点数据（以列表存储）的序号，需要根据序
        号查找到对应的节点的id，形成新的矩阵。
    第四步：
        移除sys1，重新寻找基准系统和目标系统，并重复上述三步骤。注意，在获取数据时，不
        再获取已经在上次迭代中匹配到的节点，对第三步得到的结果应该以None填充上一次迭代
        中标准系统的所在列，在本例中，为第一列：
        [[None, 0, 0, None, None],
         [None, 3, 3, 1, None],
         [None, 4, 4, None, None]]

    Args:
        label: 记录系统及其包含实体的DataFrame
        base_order(int): 基准系统的选择顺序，开始时为0
        not_extract(dict): 存储对应系统中不被抽取的节点的id

    Returns:

    """
    # 获取基准系统的信息
    if not_extract is None:
        not_extract = {}
    if label is None:
        label = LABEL
    base_sys_lab = BASE_SYS_ORDER[base_order]  # 获取基准系统的标签（空间的标签）
    level_num = 0
    while True:
        base_ent_lab = label[base_sys_lab].iloc[level_num]
        if isinstance(base_ent_lab, str):
            break
        else:
            level_num += 1
    base_pros = PRO[base_sys_lab].iloc[level_num]
    # 获取基准系统的数据
    base_data = get_data(base_sys_lab, base_ent_lab, base_pros, not_extract.get(base_sys_lab))
    if not base_data:  # 说明没有获取到基准系统的数据
        return pd.DataFrame(columns=LABEL.columns)
    if label.shape[1] == 1:  # 说明系统标签库中只剩下一个系统，不再进行融合
        return no_similarity(base_data, base_sys_lab)

    # 遍历目标系统，获取相关数据，然后逐个与基准系统进行比对
    # 并将比对的结果存储下来
    similarities = {}
    tar_sys_labs = [BASE_SYS_ORDER[i] for i in BASE_SYS_ORDER if i > base_order]
    for tar_sys_lab in tar_sys_labs:
        # 获取相同等级的实体
        tar_ent_lab = label[tar_sys_lab].iloc[level_num]
        if not isinstance(tar_ent_lab, str):
            continue
        tar_pros = PRO[tar_sys_lab].iloc[level_num]
        tar_data = get_data(tar_sys_lab, tar_ent_lab, tar_pros, not_extract.get(tar_sys_lab))
        if not tar_data:  # 说明没有获取到该目标系统的数据
            continue
        similarities[tar_sys_lab], _not_extract = compute(base_data, tar_data,
                                                          not_extract.get(tar_sys_lab))
        not_extract[tar_sys_lab] = _not_extract  # 更新不抽取的数据
    if not similarities:  # 说明遍历完所有的目标系统后均没有获取到数据，故没有融合结果
        return no_similarity(base_data, base_sys_lab)
    df = combine_sim(similarities, base_sys_lab)
    label = label.drop(base_sys_lab, axis=1)
    return df.append(fuse_root_nodes(label, base_order + 1, not_extract))


def fuse_other_nodes(level: int, node, sorted_sys: dict):
    """通过递归的方式，根据给出的根节点融合结果，对其下所有可能的节点进行融合。

    此函数没有返回值，而是将对传入的node参数进行改写，以不断挂接新的子节点。

    Args:
        level: 指定当前是第几级实体，用于控制递归的进行，0表示根节点，依次递加
        node(Nodes): 一个节点对象，存储父节点的有关信息
        sorted_sys: 配置文件中基准系统的选取序列

    Returns:

    """
    if level == LABEL.shape[0]:  # 已经到达最后一级实体
        return
    # label_df = LABEL.copy()

    # 先基于父实体的id，对其直接子节点进行完全融合
    parent_ids = node.value
    df = fuse_in_same_level(LABEL, parent_ids, level)

    if df is None:
        return
    for i in range(df.shape[0]):
        value = df.iloc[i].to_list()
        label = LABEL[sorted_sys[min(sorted_sys)]].iloc[level]
        rel = '-[:CONNECT]->'
        child = Nodes(label, value, rel)
        fuse_other_nodes(level + 1, child, sorted_sys)
        node.add_child(child)


def get_data(sys_label: str, ent_labs: str, pro_names: str, not_extract=None):
    """从图数据库获取数据。

    Args:
        sys_label: 来源系统的标签
        ent_labs: 待获取数据的实体的标签，如果以英文分号分割，则表明有多个实体标签处于同一等级
        pro_names: 待获取实体的属性名称
        not_extract: 不抽取的节点的id

    Returns:
        tuple of lists

    """
    if not_extract is None:
        not_extract = []
    graph = Graph(neo4j_url, auth=auth)
    ent_labs = ent_labs.split(';')
    pro_names = pro_names.split(';')
    # ent_lab与pro中的本体标签与融合使用的属性应该一一对应
    all_data = []
    for i in range(len(ent_labs)):
        ent_lab = ent_labs[i]
        pros = pro_names[i].split(',')
        cypher = f'match (n:{sys_label}:{ent_lab}) '
        cypher += 'return id(n) as id_, '
        for p in pros:
            cypher += 'n.' + p + f' as {p}, '
        cypher = cypher[:-2]
        data = graph.run(cypher).data()
        all_data.extend([i for i in data if i['id_'] not in not_extract])
    return all_data


def get_data2(sys_label: str, pros: list, level: int, p_node_id: int, not_extract=None, order=0):
    """根据父节点的id以及实体级别，找到其对应的所有子节点。

    Args:
        sys_label: 子实体所在系统/空间的标签
        pros: 融合子实体所依赖的属性列表
        level: 子实体的级别
        p_node_id: 父节点的id
        not_extract: 不抽取的节点的id列表
        order: 如果存在多类实体位于同一级别上，那么按照`order`指定的顺序逐个进行数据获取，
            最后再合并

    Returns:

    """
    if not_extract is None:
        not_extract = []
    graph = Graph(neo4j_url, auth=auth)
    rel = '-[:CONNECT]->'  # 父节点到此节点的关系
    tar_ent_list = LABEL[sys_label].iloc[level].split(';')
    tar_ent = tar_ent_list[order]
    cypher = f'match (n){rel}(m:`{tar_ent}`) where id(n)={int(p_node_id)} return distinct id(m) ' \
             f'as id_, m.'
    for p in pros:
        cypher += p + f' as {p}, '
    cypher = cypher[:-2]
    data = graph.run(cypher).data()
    return [i for i in data if i['id_'] not in not_extract]


def compute(base_data, tar_data, not_extract=None):
    """输入基准系统的数据和目标系统的数据，计算其中节点的相似度。

    Args:
        base_data(list[dict]): 基准系统的节点列表
        tar_data(list[dict]): 目标系统的节点列表
        not_extract(list): 不抽取的节点列表，在此处进行更新

    Returns:
        结果字典与更新后的不抽取节点列表

    """
    if not_extract is None:
        not_extract = []
    computer = Computation(threshold)
    res = computer.compute(base_data, tar_data)
    returned = {}
    if res is np.nan:
        return returned
    for i in res:
        if i[1] is not None and i[1] is not np.nan:
            returned[base_data[i[0]]['id_']] = tar_data[i[1]]['id_']
            not_extract.append(tar_data[i[1]]['id_'])
        else:
            returned[base_data[i[0]]['id_']] = np.nan
    return returned, not_extract


def combine_sim(similarities: dict, base_sys_lab: str):
    """将基准系统与目标系统的融合结果进行拼接，形成最后的融合结果。

    Args:
        similarities: 记录基准系统与目标系统之间的相似性结果，其内容格式如下：
            {
                "target_system_label_1": {
                    0: 99,
                    1: 89,
                    2: np.nan
                },
                "target_system_label_2": {
                    0: 45,
                    1: np.nan,
                    2: 34
                }
            }
            外层字典的键，是目标系统的标签；每个内层系统的键都是一样的，为基准系统中待融合的实体的id。
        base_sys_lab: 基准系统的标签

    Returns:

    """
    res = []
    base_ids = list(list(similarities.values())[0].keys())
    for id_ in base_ids:
        _l = []
        for s in LABEL.columns:
            if s == base_sys_lab:
                _l.append(id_)
                continue
            elif similarities.get(s):
                _l.append(similarities[s][id_])
            else:
                _l.append(np.nan)
        res.append(_l)
    return pd.DataFrame(data=res, columns=LABEL.columns)


def no_similarity(base_data: list, base_sys_lab: str):
    """在没有获取到目标系统标签或目标系统数据后，返回本内容"""
    res = []
    for i in base_data:
        _l = []
        for j in LABEL.columns:
            if base_sys_lab == j:
                _l.append(i['id_'])
            else:
                _l.append(np.nan)
        res.append(_l)
    return pd.DataFrame(data=res, columns=LABEL.columns)


def fuse_in_same_level(label_df: pd.DataFrame, parent_ids: list,
                       level: int, not_extract: list = None):
    """对于同一级别下的多系统实体，进行完全融合。

    运用了递归，依次寻找基准系统与目标系统。

    Args:
        label_df: 存储系统与实体标签的data frame
        not_extract: 不抽取的实体id列表，按系统名称的字典存储
        parent_ids: 存储根节点的id列表
        level: 待融合实体的层级

    Returns:

    """
    if not_extract is None:
        not_extract = {}
    systems = label_df.columns.to_list()
    labels = label_df.iloc[level].to_list()
    sorted_sys = sort_sys(label_df)  # 基准系统选择顺序
    #
    # 基准与目标系统的获取需要满足两个条件，一个是在该系统的父级融合结果中有值，另一个是
    # 系统在本次融合的级别上有实体
    #
    base_sys = ''  # 基准系统
    tar_sys_list = []  # 目标系统列表
    for i in range(len(sorted_sys)):
        base_sys = sorted_sys[i]
        if not np.isnan(parent_ids[systems.index(base_sys)]):  # 父级结果有值
            if isinstance(labels[systems.index(base_sys)], str):  # 当前级别存在实体
                tar_sys_list = list(set(sorted_sys.values()) - {base_sys})
                break
    if not base_sys:
        return None
    if not tar_sys_list:
        return None
    #
    # 获取基准系统数据
    #
    # 由于可能出现多类实体存在于同一个级别上，例如配电变压器和柱上变压器同属变压器类别
    # ，那么反映到`LABEL`, `PRO`, `TRANS`的表现就是在某个单元格内的实体标签和属性名
    # 称之间以英文分号连接。对于这种情况，融合前先将一个单元格内的所有种类的实体的数据
    # 全部获取到，然后按照相同的方法融合。
    #
    base_pros = PRO[base_sys].iloc[level].split(';')
    base_p_id = parent_ids[systems.index(base_sys)]
    if np.isnan(base_p_id):
        return None
    if len(base_pros) > 1:  # 说明出现多类实体处在同一级别上
        # 在当前的设置中，这个条件分支永远也不会被执行到
        base_data = []
        for i in range(len(base_pros)):
            base_pro = base_pros[i].split(',')
            base_data.extend(get_data2(base_sys, base_pro, level, base_p_id,
                                       not_extract.get(base_sys), i))
    else:
        base_data = get_data2(base_sys, base_pros, level, base_p_id,
                              not_extract.get(base_sys))
        if not base_data:
            return None

    # 遍历目标系统，获取相关数据，然后逐个与基准系统进行比对
    # 并将比对的结果存储下来
    similarities = {}
    for tar_sys in tar_sys_list:
        tar_pros = PRO[tar_sys].iloc[level].split(';')
        tar_p_id = parent_ids[systems.index(tar_sys)]
        if np.isnan(tar_p_id):
            continue
        if not isinstance(labels[systems.index(tar_sys)], str):
            continue
        if len(tar_pros) > 1:  # 说明出现类多类实体处在同一级别上
            tar_data = []
            for i in range(len(tar_pros)):
                tar_pro = tar_pros[i].split(',')
                tar_data.extend(get_data2(tar_sys, tar_pro, level, tar_p_id,
                                          not_extract.get(tar_sys), i))
        else:
            tar_data = get_data2(tar_sys, tar_pros, level, tar_p_id, not_extract.get(tar_sys))
            if not tar_data:
                continue
            similarities[tar_sys], _not_extract = compute(base_data, tar_data,
                                                          not_extract.get(tar_sys))
            not_extract[tar_sys] = _not_extract
    if not similarities:
        return no_similarity(base_data, base_sys)
    df = combine_sim(similarities, base_sys)
    label_df = label_df.drop(base_sys, axis=1)
    root_results_bak = parent_ids.copy()
    root_results_bak.pop(systems.index(base_sys))
    return df.append(fuse_in_same_level(label_df, root_results_bak, level, not_extract))


def create_node_and_rel(node):
    """对于一个包含子图所有信息的node，将子图生成到Neo4j中去。

    Args:
        node(Nodes): 一个`trie.Nodes`对象

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    tx = graph.begin()
    root_node = create_node(tx, node.value, node.label, 0)

    def func(p_node: Node, nodes: Nodes, i: int):
        """一个递归调用的函数。

        Args:
            p_node: 一个`py2neo.Node`对象
            nodes: 一个`trie.Nodes`对象
            i: 记录层级

        Returns:

        """
        if isinstance(nodes, list):
            return
        data = nodes.children
        for j in data:  # j也是一个`trie.Nodes`对象
            node_ = create_node(tx, j.value, j.label, i)
            rel = j.rel
            tx.create(Relationship(p_node, rel, node_))
            if not j.children:
                continue
            else:  # node_存在子节点，因此递归调用
                k = i + 1
                func(node_, j, k)

    func(root_node, node, 1)
    tx.commit()


def create_node(tx, value: list, label: str, level: int):
    """根据融合结果的列表与配置文件中的迁移属性内容，创建新的节点对象。

    Args:
        tx: 一个图数据库的事务
        value: 融合后的节点在各个系统中的值的列表，长度等于系统的数量
        label: 融合后的实体的标签，需要与`merge`合并为多标签
        level: 第几级实体

    Returns:
        一个`py2neo.Node`对象

    """
    assert len(value) == LABEL.shape[1]
    data = {}
    tran_pros = set()
    for i, v in enumerate(value):
        if np.isnan(v):
            continue
        v = int(v)
        data[f'{BASE_SYS_ORDER[i]}Id'] = v
        pros = TRANS[BASE_SYS_ORDER[i]].iloc[level].split(';')  # 某个系统、某个级别实体的迁移属性列表
        for p in pros:
            if p in tran_pros:  # 已有其他系统的属性被迁移
                continue
            else:
                val = tx.run(f"match (n) where id(n)={v} return n.{p} as p").data()[0]['p']
                if val:
                    tran_pros.add(p)
                    data[f'{p}'] = val
    return Node(*[label, fused_label], **data)


def delete_old(label):
    """Delete ole fuse results.

    Args:
        label(str): Node label

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    graph.run(f"match (n:`{label}`)-[r]-() delete r")
    graph.run(f"match (n:`{label}`) delete n")


def get_save_mapping(task_id: str):
    """获取需要存入关系库的相关数据，包括：
        - 当前计算的次数
        - 空间与本体的标签与其唯一标识及其他信息的映射字典
    """
    conn = connect(**eval(mysql_cfg))
    with conn.cursor() as cr:
        cr.execute(f"select max(`batchNo`) from gd_fuse_result where "
                   f"fuse_id='{task_id}'")
        cache = cr.fetchone()[0]
        next_batch_no = cache + 1 if cache else 1

        cr.execute(f"select space_id, space_label, ontological_id, ontological_name, "
                   f"ontological_label, ontological_weight, ontological_mapping_column_name "
                   f"from gd_fuse_attribute t where t.fuse_id = '{task_id}'")
        info = cr.fetchall()
        # info: 空间标签的唯一标识，空间标签，本体标签的唯一标识，本体名称，
        #       本体标签，本体权重，融合字段
    mapping = {}
    for i in info:
        mapping[i[1]] = i[0]
        mapping[i[4]] = [i[2], i[3], i[5], i[6]]

    return mapping, next_batch_no


def caching(counter_only: pd.DataFrame, counter_all: pd.DataFrame, node: Nodes, level=0):
    """对于每一个子图，计算其中的每个空间下的不同本体中的融合统计情况，包括：
        - 独有的数量
        - 共有的数量
    """
    if node.value.count(None) == len(node.value) - 1:
        for i in range(len(node.value)):
            if i is not None:
                counter_only.iloc[level, i] += 1
                break
    else:
        for i in range(len(node.value)):
            if i is not None:
                counter_all.iloc[level, i] += 1
    if node.children:
        for n in node.children:
            a, b = caching(counter_only, counter_all, n, level+1)
            counter_only.append(a)
            counter_all.append(b)
    return counter_only, counter_all


def save_res_to_mysql(counter_only, counter_all, mapping, next_batch_no, task_id, start_time):
    """将融合的统计结果写入关系型数据库"""
    conn = connect(**eval(mysql_res))
    end_time = strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cr:
        for i in range(counter_only.shape[0]):
            for j in range(counter_only.shape[1]):
                id_ = str(uuid1())
                space_id = mapping[counter_only.columns[j]]
                label = LABEL.iloc[i, j]
                ontological_id, ontological_name, ontological_weight, merge_cols = mapping[label]
                matched = counter_all.iloc[i, j]
                only = counter_only.iloc[i, j]
                sql = f"insert into gd_fuse_result values ('{id_}', '{task_id}', '{space_id}', " \
                      f"'{ontological_id}', '{ontological_name}', '{label}', " \
                      f"{ontological_weight}, '{merge_cols}',{matched}, {only}, {next_batch_no}, " \
                      f"'{start_time}', '{end_time}')"
                cr.execute(sql)
    conn.commit()
