# -*- coding: utf-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Main functions to run fusing tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
from configparser import ConfigParser
from utils import Nodes, sort_sys, Computation
from self_check import get_paras
from pymysql import connect
from time import strftime
from uuid import uuid1
import requests
from log_utils import gen_logger
from os.path import split, abspath

logger = None
LABEL, PRO, TRANS = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
fused_label = ''
BASE_SYS_ORDER = {}
cfg = ConfigParser()
current_path = split(abspath(__file__))[0]
with open(current_path + '/config_files/application.cfg', encoding='utf-8') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
threshold = float(cfg.get('threshold', 'threshold'))
mysql_res = cfg.get('mysql', 'mysql_res')
mysql_cfg = cfg.get('mysql', 'mysql_cfg')
err_url = cfg.get('error_handler', 'url')


def main_fuse(task_id: str) -> None:
    """Main function to run the fusion task.

    The application receives an HTTP request as a start and then go to this function to
    do the calculation.

    This function has no return value and will send data, which is `dict` type in
    Python, to specific url when needed. The keys of sent data are within "task_id",
    "state", "msg", "progress", whose explanation are as follows::

        "task_id": The unique identification of task.
        "state": The state of task, and its possible values are 0(task ended normally),
            1(some errors occur) and 2(task is in progress).
        "msg": Message that helps to understand what is going on.
        "progress": This tells the progress of task calculation when "state" is 2.

    Args:
        task_id: Unique identification of task

    Returns:
        None

    """
    global logger
    logger = gen_logger(task_id)
    global LABEL, PRO, TRANS, BASE_SYS_ORDER, fused_label
    LABEL, PRO, TRANS, fused_label = get_paras(task_id)

    BASE_SYS_ORDER = sort_sys(LABEL)
    logger.info("Start to fuse...")
    logger.info("Deleting old results..")
    delete_old(fused_label)
    logger.info("Deletion complete")
    logger.info("Fusing root nodes...")
    start_time = strftime("%Y-%m-%d %H:%M:%S")
    mapping, next_batch_no = get_save_mapping(task_id)
    # counter_only = pd.DataFrame(data=0, columns=LABEL.columns, index=LABEL.index)
    # counter_all = pd.DataFrame(data=0, columns=LABEL.columns, index=LABEL.index)
    try:
        root_res_df = fuse_root_nodes()
    except Exception as e:
        logger.info(e)
        err_data = {
            "task_id": task_id,
            "state": "1",
            "msg": "Errors occur in root nodes fusion",
            "progress": 0
        }
        requests.post(err_url, json=err_data)
        return
    else:
        if root_res_df.empty:
            logger.info("Root nodes fusion result is empty")
            err_data = {
                "task_id": task_id,
                "state": "0",
                "msg": "Root nodes fusion result is empty",
                "progress": 0
            }
            requests.post(err_url, json=err_data)
            return
        else:
            logger.info("Root nodes fusion complete, start to fuse children nodes...")
            try:
                base_ent_lab = LABEL[BASE_SYS_ORDER[0]].iloc[0]
                for i in range(len(root_res_df)):
                    progress_data = {
                        "task_id": task_id,
                        "state": 2,
                        "msg": "Fusing children nodes...",
                        "progress": (i + 1) / len(root_res_df)
                    }
                    node = Nodes(base_ent_lab, root_res_df.iloc[i].to_list())
                    fuse_other_nodes(1, node,
                                     BASE_SYS_ORDER)  # 执行之后，node包含了创建一个子图所需要的完整信息
                    # a, b = caching(counter_only, counter_all, node)
                    # counter_only.append(a)
                    # counter_all.append(b)
                    stat_info = create_node_and_rel(node)
                    _ = requests.post(err_url, json=progress_data)
                # save_res_to_mysql(counter_only, counter_all, mapping, next_batch_no,
                #                   task_id, start_time)
                    save_res_to_mysql2(stat_info, mapping, next_batch_no, task_id, start_time)
                logger.info("Fusion graph creation complete")
                finish_data = {
                    "task_id": task_id,
                    "state": 0,
                    "msg": "Complete",
                    "progress": 1
                }
                requests.post(err_url, json=finish_data)
            except Exception as e:
                logger.info(e)
                err_data = {
                    "task_id": task_id,
                    "state": "1",
                    "msg": "Errors occur in children nodes fusion",
                    "progress": 0
                }
                requests.post(err_url, json=err_data)
                return


def fuse_root_nodes(label: pd.DataFrame = None, base_order=0, not_extract=None):
    """Function to fuse root nodes.

    **Root node** is defined as the very first entity class of each system. In future
    versions, more information, such as administrative divisions, may be added to the
    code to speed up fusion.

    The fusion progress will be executed recursively if there are over two systems to
    be fused, which means program will find a *base system* first and the rest are
    regarded as *target systems*, then traverse all *target systems* to compute
    similarities with *base system*. This is called finishing one operation. Next step
    will repeat above procedures among *target systems*.

    The following is an example:

    Suppose there are 5 systems that need to be fused:

    +------+-------+-------+-------+-------+-------+
    |      |sys1   |sys2   |sys3   |sys4   |sys5   |
    +======+=======+=======+=======+=======+=======+
    |level1| Ent   | Ent   | Ent   | Ent   | Ent   |
    +------+-------+-------+-------+-------+-------+
    |level2| Ent   | Ent   | Ent   | Ent   | Ent   |
    +------+-------+-------+-------+-------+-------+
    |...   | ...   | ...   | ...   | ...   | ...   |
    +------+-------+-------+-------+-------+-------+

    .. Note::
       Only *level1* entities' labels of each systems are needed when fusing root nodes.

    * Step 1:
        Assume that ``sys1`` is the *base system* and others are *target systems* in
        the first level of recursion, what we need to do is calculating the
        similarities between nodes in *base system* and nodes in *target systems*
        separately. In order to saving time, nodes are determined as the same
        automatically if they are come from different *target systems* and fused  with
        one node in *base system* during the computation.

    * Step 2:
        The fused results that come from *base system* and *target systems* contain all
        nodes' ids in *base system* while not all nodes' ids from *target systems*. For
        example, ``sys1`` is *base system* and have 3 nodes, ``sys2`` is *target system*
        and have 4 nodes. Then fused results could be ``[[0, 1], [1, None], [2, 2]]``.

    * Step 3:
        Combine the preliminary results of each two systems to get the final results of
        first recursion, which may have the following format::

            [[0, 1, None, None, 1],
             [1, None, 1, 2, 0],
             [2, 2, 2, 0, 2]]

        Note that integers in results list is node index in node list that read from
        `Neo4j`, not nodes' id.

    * Step 4:
        Remove first *base system*, ``sys1`` in this example, and execute programs to
        find new *base system* and *target systems* then repeat the above three steps.
        Note that nodes which already get matched in previous computation won't be
        extracted in next recursion. Besides, ``None`` will be padded to result list at
        the position(s) of previous *base system(s)*. In the second recursion of this
        case, it's the first 'column'::

            [[None, 0, 0, None, None],
             [None, 3, 3, 1, None],
             [None, 4, 4, None, None]]

    Args:
        label: A dataframe that contains system labels and their
            entity labels
        base_order(int): Selection order of base systems
        not_extract(dict): A dict contains the nodes' ids of each system that won't be
            extracted

    Returns:
        `pandas.DataFrame`

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
    base_data = get_data(base_sys_lab, base_ent_lab, base_pros,
                         not_extract.get(base_sys_lab))
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
        tar_data = get_data(tar_sys_lab, tar_ent_lab, tar_pros,
                            not_extract.get(tar_sys_lab))
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
    """Fuse all children nodes recursively basing on root nodes result.

    Note that this function has no returns, and it keeps rewriting the ``node``
    variable during the recursions to add new fused children nodes.

    Args:
        level: Entity level, used to control recursion
        node(Nodes): A :py:meth:`~utils.Nodes` object to store fused information
        sorted_sys: A dict contains sorted systems according to sorting algorithm

    Returns:
        None

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
    """Get data from `Neo4j` to fuse root nodes.

    Args:
        sys_label: System label of data
        ent_labs: Entity label of data, if it's separated by ';' that means there are
            multiple entities in the same level
        pro_names: Property names that needed to be extracted
        not_extract: Nodes' ids that not to be extracted

    Returns:
        A list of lists, each inner list represents a node in `Neo4j`.

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


def get_data2(sys_label: str, pros: list, level: int, p_node_id: int, not_extract=None,
              order=0):
    """Get data from `Neo4j` according to parent node's id.

    Args:
        sys_label: System label of data
        pros: Property names that needed to be extracted
        level: Data level
        p_node_id: Parent node's id
        not_extract: Nodes' ids that not to be extracted
        order: If multiple entities in the same level, then obtain data one by one  in the
            order specified by `order` and combine together later

    Returns:
        A list of lists, each inner list represents a node in `Neo4j`.

    """
    if not_extract is None:
        not_extract = []
    graph = Graph(neo4j_url, auth=auth)
    rel = '-[:CONNECT]->'  # 父节点到此节点的关系
    tar_ent_list = LABEL[sys_label].iloc[level].split(';')
    tar_ent = tar_ent_list[order]
    cypher = f'match (n){rel}(m:`{tar_ent}`) where id(n)={int(p_node_id)} return ' \
             f'distinct id(m) ' \
             f'as id_, m.'
    for p in pros:
        cypher += p + f' as {p}, '
    cypher = cypher[:-2]
    data = graph.run(cypher).data()
    return [i for i in data if i['id_'] not in not_extract]


def compute(base_data, tar_data, not_extract=None):
    """Compute the similarities of the given data.

    The similarity results will be stored as a dict, whose key is node's id in
    `base_data` and corresponding value is **the most similar** node's id in `tar_data`.

    Args:
        base_data(list[dict]): A list of dicts, each dict represents a node of *base
            system*
        tar_data(list[dict]): A list of dicts, each dict represents a node of *target
            system*
        not_extract(list): Nodes' ids that not to be extracted, and will be updated after
            computation

    Returns:
        A dict of similarity results and updated `not_extract` list.

    """
    if not_extract is None:
        not_extract = []
    computer = Computation(threshold)
    res = computer.compute(base_data, tar_data)
    returned = {}
    if res is np.nan:
        return returned
    for i in res:
        if i[0] is None:
            continue
        if i[1] is not None and i[1] is not np.nan:
            returned[base_data[i[0]]['id_']] = tar_data[i[1]]['id_']
            not_extract.append(tar_data[i[1]]['id_'])
        else:
            returned[base_data[i[0]]['id_']] = np.nan
    return returned, not_extract


def combine_sim(similarities: dict, base_sys_lab: str):
    """Transform similarity dict into a `pandas.DataFrame` object.

    Args:
        similarities: A dict of similarity results of *base system* to *target systems*,
            whose format is as followed::

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

            Keys of outer dict are labels of *target systems* and the inner dicts have
            the same structure, keys of which are nodes' ids of *base system*.
        base_sys_lab: Label of *base system*

    Returns:
        A `pandas.DataFrame` stands for the similarity results.

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
    """This function is used when obtain no data from one of *target systems*.

    Args:
        base_data: Data of *base system*
        base_sys_lab: Label of *base system*

    Returns:
        A `pandas.DataFrame` represents similarity results(only *base data* ).

    """
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
                       level: int, not_extract: dict = None):
    """Fuse all entities under one group of parent ids using recursive methods.

    Args:
        label_df: A `pandas.DataFrame` to store system labels and entity labels and has
            the same structure with `LABEL`
        not_extract: A dict contains the lists of nodes' ids that not to be extracted
        parent_ids: Root nodes' ids
        level: Entity level

    Returns:
        The final return is a `pandas.DataFrame`, but in some certain conditions it will
        return `None`.

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
            tar_data = get_data2(tar_sys, tar_pros, level, tar_p_id,
                                 not_extract.get(tar_sys))
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
    """Create a subgraph into `Neo4j` according to `node`.

    Args:
        node(Nodes): A :py:meth:`~utils.Nodes` object

    Returns:
        None

    """
    graph = Graph(neo4j_url, auth=auth)
    tx = graph.begin()
    root_node = create_node(tx, node.value, node.label, 0)

    stat_info = {}  # 统计信息

    def statistic(a_neo_node, i):
        dict_info = dict(a_neo_node.items())
        label = str(a_neo_node.labels).split(':')
        label.remove('')
        label.remove(fused_label)  # 本体的标签
        space = BASE_SYS_ORDER[i]  # 空间的标签
        num = 0
        for k in BASE_SYS_ORDER:
            if dict_info.get(f'{BASE_SYS_ORDER[k]}Id'):
                num += 1
        if space + '/' + label not in stat_info:
            if num > 1:
                if dict_info.get('orgno'):
                    stat_info[space+'/'+label] = {dict_info['orgno']: {'all': 1, 'only': 0}}
                stat_info[space+'/'+label] = {'all': 1, 'only': 0}
            else:
                if dict_info.get('orgno'):
                    stat_info[space+'/'+label] = {dict_info['orgno']: {'all': 0, 'only': 1}}
                stat_info[space+'/'+label] = {'all': 0, 'only': 1}
        else:
            if num > 1:
                if dict_info.get('orgno'):
                    stat_info[space+'/'+label][dict_info['orgno']]['all'] += 1
                stat_info[space+'/'+label]['all'] += 1
            else:
                if dict_info.get('orgno'):
                    stat_info[space+'/'+label][dict_info['orgno']]['only'] += 1
                stat_info[space+'/'+label]['only'] += 1

    statistic(root_node, 0)

    def func(p_node: Node, nodes: Nodes, i: int):
        """A recursive function to create subgraph.

        Args:
            p_node: A `py2neo.Node` object, represents parent node which is already
                created but not committed
            nodes: A :py:meth:`~utils.Nodes` object
            i: level of depth

        Returns:
            None

        """
        if isinstance(nodes, list):
            return
        data = nodes.children
        if not data:
            statistic(p_node, i)
            tx.create(p_node)
            return
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
    return stat_info


def create_node(tx, value: list, label: str, level: int):
    """Create the new node according to a fuse result list.

    The new node needs some properties and the corresponding values, which are transferred
    from original nodes.

    Args:
        tx: A `Neo4j` transaction object
        value: A list of original nodes' ids
        label: The new label of the new fused nodes
        level: Entity level

    Returns:
        A :py:meth:`~py2neo.Node` object

    """
    assert len(value) == LABEL.shape[1]
    data = {}
    tran_pros = set()
    for i, v in enumerate(value):
        if np.isnan(v):
            continue
        v = int(v)
        data[f'{BASE_SYS_ORDER[i]}Id'] = v
        org_no = tx.run(f"match(n) where id(n)={v} return n.orgno as orgno").data()[0]['orgno']
        data[f'{BASE_SYS_ORDER[i]}OrgNo'] = org_no

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
    """Delete old fuse results.

    Args:
        label(str): Nodes' label

    Returns:
        None

    """
    graph = Graph(neo4j_url, auth=auth)
    graph.run(f"match (n:`{label}`)-[r]-() delete r")
    graph.run(f"match (n:`{label}`) delete n")


def get_save_mapping(task_id: str):
    """Obtain the data that needs to be stored into `MySQL`. Including:

    - Times of computation for the same task
    - Labels of system and ontology and some other mapping information

    Args:
        task_id: Id of fuse task

    Returns:
        A dict contains mapping information and an integer stands for times of
        computation.

    """
    conn = connect(**eval(mysql_cfg))
    with conn.cursor() as cr:
        cr.execute(f"select max(`batchNo`) from gd_fuse_result where "
                   f"fuse_id='{task_id}'")
        cache = cr.fetchone()[0]
        next_batch_no = cache + 1 if cache else 1

        cr.execute(f"select space_id, space_label, ontological_id, ontological_name, "
                   f"ontological_label, ontological_weight, "
                   f"ontological_mapping_column_name "
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
    """Count the number of each ontology in subgraphs, including:

    - Number of ontology that exists in only one system
    - Number of ontology that exists in more than one system

    Args:
        counter_only: A `pandas.DataFrame` that stores the number of ontology only exists
            in ont system
        counter_all: A `pandas.DataFrame` that stores the number of ontology exists in
            more than one system
        node: A :py:meth:`~utils.Nodes` object
        level: Entity level

    Returns:
        `counter_only` and `counter_all`.

    """
    if pd.Series(node.value).count() == 1:
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
            a, b = caching(counter_only, counter_all, n, level + 1)
            counter_only.append(a)
            counter_all.append(b)
    return counter_only, counter_all


def save_res_to_mysql(counter_only, counter_all, mapping, next_batch_no, task_id,
                      start_time):
    """Save the count results to `MySQL`"""
    conn = connect(**eval(mysql_res))
    end_time = strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cr:
        for i in range(counter_only.shape[0]):
            for j in range(counter_only.shape[1]):
                id_ = str(uuid1())
                space_id = mapping[counter_only.columns[j]]
                label = LABEL.iloc[i, j]
                ontological_id, ontological_name, ontological_weight, merge_cols = \
                    mapping[label]
                matched = counter_all.iloc[i, j]
                only = counter_only.iloc[i, j]
                sql = f"insert into gd_fuse_result values ('{id_}', '{task_id}', " \
                      f"'{space_id}', '{ontological_id}', '{ontological_name}', " \
                      f"'{label}',{ontological_weight}, '{merge_cols}',{matched}, " \
                      f"{only}, {next_batch_no}, '{start_time}', '{end_time}')"
                cr.execute(sql)
    conn.commit()


def save_res_to_mysql2(stat_info, mapping, next_batch_no, task_id, start_time):
    conn = connect(**eval(mysql_res))
    end_time = strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cr:
        for i in stat_info:
            space, ontology = i.split('/')
            id_ = str(uuid1())
            space_id = mapping[space]
            ontological_id, ontological_name, ontological_weight, merge_cols = \
                mapping[ontology]
            total_matched, total_only = stat_info[i]['all'], stat_info[i]['only']
            cr.execute(f"insert into gd_fuse_result values ('{id_}', '{task_id}', "
                       f"'{space_id}', '{ontological_id}', '{ontological_name}', "
                       f"'{ontology}',{ontological_weight}, '{merge_cols}',{total_matched}, "
                       f"{total_only}, {next_batch_no}, '{start_time}', '{end_time}', '{None}')")
            id_ = str(uuid1())
            for j in stat_info[i]:
                if j not in ['all', 'only']:
                    matched, only = stat_info[i][j]['all'], stat_info[i][j]['only']
                    sql = f"insert into gd_fuse_result values ('{id_}', '{task_id}', " \
                          f"'{space_id}', '{ontological_id}', '{ontological_name}', " \
                          f"'{ontology}',{ontological_weight}, '{merge_cols}',{matched}, " \
                          f"{only}, {next_batch_no}, '{start_time}', '{end_time}', '{j}')"
                    cr.execute(sql)
    conn.commit()
