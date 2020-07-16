# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : fuse.py                         #
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

问题记录：
    - 嵌套递归的效率问题。
    - 按照当前的处理逻辑，基准系统的的一级实体必须存在，否则会处理出错。
      出错代码在``fuse_root_nodes``中，变量`base_ent_lab`的值会变为`np.nan`。这是由于基准系统选择算法
      只考虑了该系统纳入融合计算的所有的实体类别的数量，并未考察一级实体是否有值。应该修改基准系统选择算法。

"""
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
from configparser import ConfigParser
from text_sim_utils import sims
from trie import Nodes

format_ = pd.read_excel('./config_files/format.xlsx',
                        sheet_name=['label', 'rel', 'pro', 'trans'], index_col=0)
LABEL, REL, PRO, TRANS = format_.values()
LABEL.replace(r'^\s+$', np.nan, regex=True, inplace=True)
REL.replace(r'^\s+$', np.nan, regex=True, inplace=True)
PRO.replace(r'^\s+$', np.nan, regex=True, inplace=True)
TRANS.replace(r'^\s+$', np.nan, regex=True, inplace=True)

cfg = ConfigParser()
with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
threshold = float(cfg.get('threshold', 'threshold'))


def sort_sys(label: pd.DataFrame) -> dict:
    """根据配置文件的系统与实体标签，计算其中的基准系统，按照顺序进行排列"""
    res = {}
    label_bak = label.copy()
    for i in range(label.shape[1]):
        lab = label_bak.columns.values[label_bak.count().argmax()]
        res[i] = lab
        label_bak.drop(lab, axis=1, inplace=True)
    return res


def fuse_root_nodes(label: pd.DataFrame, pro: pd.DataFrame, not_extract=None):
    """对根节点进行融合。

    根节点指的是各个系统中的第一个实体类，为了加快融合速度，应该增加额外的信息以
    帮助程序缩小检查范围，如行政区域划分等。

    融合过程需要递归进行，即逐层获取基准系统与目标系统进行相似度计算，获取融合结果，
    最后再将融合结果统一整理。

    假设有5个系统需要融合：
              | sys1 | sys2 | sys3 | sys4 | sys5 |
        ------|------|------|------|------|------|
        1级实体| Ent  | Ent  | Ent  | Ent  | Ent  |
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
        pro: 记录系统及其可供融合利用的实体属性的DataFrame
        not_extract(dict): 存储对应系统中不被抽取的节点的id

    Returns:

    """
    sort_res = sort_sys(label)
    # 获取基准系统的信息
    if not_extract is None:
        not_extract = {}
    base_sys_lab = sort_res[min(sort_res)]
    base_ent_lab = label[base_sys_lab].iloc[0]
    base_pros = pro[base_sys_lab].iloc[0].split(',')
    # 获取基准系统的数据
    base_data = get_data(base_sys_lab, base_ent_lab,
                         base_pros, not_extract.get(base_sys_lab))
    if not base_data:  # 说明没有获取到基准系统的数据
        return pd.DataFrame(columns=LABEL.columns)
    if label.shape[1] == 1:  # 说明系统标签库中只剩下一个系统，不再进行融合
        return no_similarity(base_data, base_sys_lab)

    # 遍历目标系统，获取相关数据，然后逐个与基准系统进行比对
    # 并将比对的结果存储下来
    similarities = {}
    tar_sys_labs = list(set(sort_res.values()) - {base_sys_lab})
    for tar_sys_lab in tar_sys_labs:
        tar_ent_lab = label[tar_sys_lab].iloc[0]
        tar_pros = pro[tar_sys_lab].iloc[0].split(',')
        tar_data = get_data(tar_sys_lab, tar_ent_lab,
                            tar_pros, not_extract.get(tar_sys_lab))
        if not tar_data:  # 说明没有获取到该目标系统的数据
            continue
        similarities[tar_sys_lab], _not_extract = compute(base_data, tar_data,
                                                          not_extract.get(tar_sys_lab))
        not_extract[tar_sys_lab] = _not_extract  # 更新不抽取的数据
    if not similarities:  # 说明遍历完所有的目标系统后均没有获取到数据，故没有融合结果
        return no_similarity(base_data, base_sys_lab)
    df = combine_sim(similarities, base_sys_lab)
    label = label.drop(base_sys_lab, axis=1)
    return df.append(fuse_root_nodes(label, pro, not_extract))


def fuse_other_nodes(start_index: int, node, sorted_sys: dict):
    """通过递归的方式，根据给出的根节点融合结果，对其下所有可能的节点进行融合。

    此函数没有返回值，而是将对传入的node参数进行改写，以不断挂接新的子节点。

    Args:
        start_index: 指定当前是第几级实体，用于控制递归的进行，0表示根节点，依次递加
        node(Nodes): 一个节点对象，存储父节点的有关信息
        sorted_sys: 配置文件中基准系统的选取序列

    Returns:

    """
    if start_index == LABEL.shape[0]:  # 已经到达最后一级实体
        return
    label_df = LABEL.copy()

    # 先基于父实体的id，对其直接子节点进行完全融合
    root = node.value
    df = fuse_in_same_level(label_df, root, start_index)

    if df is None:
        return
    for i in range(df.shape[0]):
        value = df.iloc[i].to_list()
        label = LABEL[sorted_sys[min(sorted_sys)]].iloc[start_index]
        rel = REL[sorted_sys[min(sorted_sys)]].iloc[start_index-1]
        child = Nodes(label, value, rel)
        fuse_other_nodes(start_index + 1, child, sorted_sys)
        node.add_child(child)


def get_data(system: str, ent_lab: str, pro, not_extract=None):
    """从图数据库获取数据。

    Args:
        system: 来源系统的标签
        ent_lab: 待获取数据的实体的标签
        pro: 待获取实体的属性名称，str、list
        not_extract: 不抽取的节点的id

    Returns:
        tuple of lists

    """
    if not_extract is None:
        not_extract = []
    graph = Graph(neo4j_url, auth=auth)
    if isinstance(pro, str):
        pro = [pro]
    cypher = f'match (n:{system}:{ent_lab}) return id(n) as id_, n.'
    for p in pro:
        cypher += p + f' as {p}, '
    cypher = cypher[:-2]
    data = graph.run(cypher).data()
    return [i for i in data if i['id_'] not in not_extract]


def get_data2(system: str, pro, level: int, p_node_id: int, not_extract=None):
    """根据父节点的id以及实体级别，找到其对应的所有子节点。

    Args:
        system: 来源系统的标签
        pro: 待获取的实体的属性，str/list
        level: p_node_id所在的实体级别
        p_node_id: 父节点的id
        not_extract: 不抽取的节点的id列表

    Returns:

    """
    if not_extract is None:
        not_extract = []
    graph = Graph(neo4j_url, auth=auth)
    if isinstance(pro, str):
        pro = [pro]
    rel = REL[system].iloc[level]  # 父节点到此节点的关系
    tar_sys = LABEL[system].iloc[level+1]  # 子节点的实体标签
    cypher = f'match (n){rel}(m:`{tar_sys}`) where id(n)={int(p_node_id)} return distinct id(m) ' \
             f'as id_, m.'
    for p in pro:
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
        if i[1] is not None:
            returned[base_data[i[0]]['id_']] = tar_data[i[1]]['id_']
            not_extract.append(tar_data[i[1]]['id_'])
        else:
            returned[base_data[i[0]]['id_']] = np.nan
    return returned, not_extract


def combine_sim(similarities: dict, base_sys_lab: str):
    """将基准系统与目标系统的融合结果进行拼接，形成最后的融合结果。

    Args:
        similarities: 记录基准系统与目标系统之间的相似性结果
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


class Computation:
    """根据从neo4j中读取得到的数据，计算相似度"""

    def __init__(self, thresh=0.75):
        """判定为同一对象的阈值，默认为0.75，可以通过配置文件修改"""
        self.thresh = thresh

    def compute(self, base_data, tar_data):
        """对外提供的计算相似度的方法接口。

        Args:
            base_data(list[dict]): 字典组成的列表，每个字典都表示一个节点
            tar_data(list[dict]): 字典组成的列表，每个字典都表示一个节点

        Returns:
            None/List

        """
        sim = np.zeros(shape=(len(base_data), len(tar_data)))
        for i in range(len(base_data)):
            for j in range(len(tar_data)):
                sim[i, j] = self.__compute(base_data[i], tar_data[j])
        return self.__matching(sim)

    @staticmethod
    def __compute(dict1, dict2) -> float:
        """计算两个节点的相似度。

        Args:
            dict1(dict): 表示一个节点的字典
            dict2(dict): 表示一个节点的字典

        Returns:
            相似度的值

        """
        sim = 0
        weight = 1 / (len(dict1) - 1)
        for k in dict1:
            if k == 'id_':
                continue
            else:
                sim += weight * sims(dict1[k], dict2[k])
        return sim

    def __match(self, sim_matrix):
        """从相似度矩阵里面选择最相似的实体对，采用了递归的方法。

        注意，由于`numpy.array.argmax()`方法在存在多个最大值的情况下默认只返回第一个的索引，
        这种特性在某些情况下可能会导致错误的融合结果。

        Args:
            sim_matrix: 相似度矩阵

        Returns:
            None或者一个嵌套的列表

        """
        if np.sum(sim_matrix) == 0:
            return None
        res = []
        args0 = sim_matrix.argmax(axis=0)  # 每一列的最大值位置
        args1 = sim_matrix.argmax(axis=1)  # 每一行的最大值位置
        for i in range(len(args1)):
            if sim_matrix[i, args1[i]] < self.thresh:
                sim_matrix[i, :] = 0
                if [i, None] not in res:
                    res.append([i, None])
            else:
                if args0[args1[i]] == i:
                    res.append([i, args1[i]])
                    sim_matrix[i, :] = 0
                    sim_matrix[:, args1[i]] = 0

        r = self.__match(sim_matrix)
        if r:
            return res + r
        else:
            return res

    def __matching(self, sim_matrix):
        """在调用`self.__match`之后，在横向或者纵向上可能遗留下一些非0值，对这些值
        进行处理。

        只处理基准表对应的数据，以保证基准表中所有的节点都被计算完成。

        Args:
            sim_matrix: 相似度矩阵

        Returns:
            嵌套的列表

        """
        res = self.__match(sim_matrix)
        if res is None:
            return np.nan
        x = set([i[0] for i in res])
        # y = set([i[1] for i in res])
        for i in range(sim_matrix.shape[0]):
            if i not in x:
                res.append([i, np.nan])
        # for j in range(sim_matrix.shape[1]):
        #     if j not in y:
        #         res.append([None, j])
        return res


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


def fuse_in_same_level(label_df: pd.DataFrame, root_results: list,
                       start_index: int, not_extract: list = None):
    """对于同一级别下的多系统实体，进行完全融合。

    运用了递归，依次寻找基准系统与目标系统。

    Args:
        label_df: 存储系统与实体标签的data frame
        not_extract: 不抽取的实体id列表，按系统名称的字典存储
        root_results: 存储根节点的id列表
        start_index: 待融合实体的层级

    Returns:

    """
    if not_extract is None:
        not_extract = {}
    systems = label_df.columns.to_list()
    labels = label_df.iloc[start_index].to_list()
    sorted_sys = sort_sys(label_df)  # 基准系统选择顺序
    #
    # 基准与目标系统的获取需要满足两个条件，一个是在该系统的父级融合结果中有值，另一个是
    # 系统在本次融合的级别上有实体
    #
    base_sys = ''
    tar_sys_list = []
    for i in range(len(sorted_sys)):
        base_sys = sorted_sys[i]
        if not np.isnan(root_results[systems.index(base_sys)]):
            if isinstance(labels[systems.index(base_sys)], str):
                tar_sys_list = list(set(sorted_sys.values()) - {base_sys})
                break
    if not base_sys:
        return None
    if not tar_sys_list:
        return None

    # 获取基准系统数据
    base_pros = PRO[base_sys].iloc[start_index].split(',')
    base_p_id = root_results[systems.index(base_sys)]
    if np.isnan(base_p_id):
        return None
    base_data = get_data2(base_sys, base_pros, start_index - 1, base_p_id,
                          not_extract.get(base_sys))
    if not base_data:
        return None

    # 遍历目标系统，获取相关数据，然后逐个与基准系统进行比对
    # 并将比对的结果存储下来
    similarities = {}
    for tar_sys in tar_sys_list:
        tar_pros = PRO[tar_sys].iloc[start_index].split(',')
        tar_p_id = root_results[systems.index(tar_sys)]
        if np.isnan(tar_p_id):
            continue
        if not isinstance(labels[systems.index(tar_sys)], str):
            continue
        tar_data = get_data2(tar_sys, tar_pros, start_index - 1, tar_p_id, not_extract.get(tar_sys))
        if not tar_data:
            continue
        similarities[tar_sys], _not_extract = compute(base_data, tar_data,
                                                      not_extract.get(tar_sys))
        not_extract[tar_sys] = _not_extract
    if not similarities:
        return no_similarity(base_data, base_sys)
    df = combine_sim(similarities, base_sys)
    label_df = label_df.drop(base_sys, axis=1)
    root_results_bak = root_results.copy()
    root_results_bak.pop(systems.index(base_sys))
    return df.append(fuse_in_same_level(label_df, root_results_bak, start_index, not_extract))


def create_node_and_rel(node):
    """对于一个包含子图所有信息的node，将子图生成到Neo4j中去。

    Args:
        node(Nodes): 一个`trie.Nodes`对象

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    tx = graph.begin()
    root_node = create_node(node.value, node.label, 0)
    # tx.create(root_node)

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
            node_ = create_node(j.value, j.label, i)
            rel = j.rel
            tx.create(Relationship(p_node, rel, node_))
            if not j.children:
                continue
            else:  # node_存在子节点，因此递归调用
                k = i + 1
                nodes = j.children
                func(node_, nodes, k)

    func(root_node, node, 1)
    tx.commit()


def create_node(value: list, label: str, level: int):
    """根据融合结果的列表与配置文件中的迁移属性内容，创建新的节点对象。

    Args:
        value: 融合后的节点在各个系统中的值的列表，长度等于系统的数量
        label: 融合后的实体的标签，需要与`merge`合并为多标签
        level: 第几级实体

    Returns:
        一个`py2neo.Node`对象

    """
    assert len(value) == LABEL.shape[1]
    data = {}
    graph = Graph(neo4j_url, auth=auth)
    sorted_sys = sort_sys(LABEL)
    tran_pros = set()
    for i, v in enumerate(value):
        if np.isnan(v):
            continue
        v = int(v)
        data[f'{sorted_sys[i]}Id'] = v
        pros = TRANS[sorted_sys[i]].iloc[level].split(',')  # 某个系统、某个级别实体的迁移属性列表
        for p in pros:
            if p in tran_pros:  # 已有其他系统的属性被迁移
                continue
            else:
                val = graph.run(f"match (n) where id(n)={v} return n.{p} as p").data()[0]['p']
                if val:
                    tran_pros.add(p)
                    data[f'{p}'] = val
    return Node(*[label, 'merge'], **data)


def delete_old(label):
    """Delete ole fuse results.

    Args:
        label(str): Node label

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    graph.run(f"match (n:`{label}`)-[r]-() delete r")
    graph.run(f"match (n:`{label}`) delete n")

# if __name__ == '__main__':
    # df = fuse_root_nodes(LABEL, PRO, sort_sys(LABEL))
    # print(df)
    # node = Nodes("Subs", [64669, 82334, 52556])
    # # fuse_other_nodes(1, node, sort_sys(LABEL))
    # print(create_node(node.value, 'Subs', 0))
