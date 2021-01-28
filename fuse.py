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
import traceback
from collections import OrderedDict

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
    LABEL, PRO, TRANS, fused_label, new_batch = get_paras(task_id)
    print(LABEL)
    print(PRO)
    BASE_SYS_ORDER = sort_sys(LABEL)
    logger.info("Start to fuse...")
    logger.info("Fusing root nodes...")
    start_time = strftime("%Y-%m-%d %H:%M:%S")
    try:
        root_res_df = fuse_root_nodes()
    except Exception as e:
        logger.info(e)
        logger.info(traceback.print_exc())
        return
    else:
        if root_res_df.empty:
            logger.info("Root nodes fusion result is empty")
            return
        else:
            logger.info("Root nodes fusion complete, start to fuse children nodes...")
            try:
                base_ent_lab = LABEL[BASE_SYS_ORDER[0]].iloc[0]
                for i in range(len(root_res_df)):
                    logger.info(f'进度：{i}/{len(root_res_df)}')
                    update_progress(task_id, int(100 * (i + 1)/len(root_res_df)))
                    node = Nodes(base_ent_lab, root_res_df.iloc[i].to_list())
                    fuse_other_nodes(1, node, BASE_SYS_ORDER)  # 执行之后，node包含了创建一个子图所需要的完整信息
                    save_detail_to_mysql(node, task_id, new_batch)
                logger.info("Fusion graph creation complete")
                update_status(start_time, task_id)
                finish_data = {
                    "task_id": task_id,
                    "state": 0,
                    "msg": "Complete",
                    "progress": 1
                }
                requests.post(err_url, json=finish_data)
            except Exception as e:
                logger.error(traceback.print_exc())
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
        # print(base_data)
        # print(tar_data)
        similarities[tar_sys_lab], _not_extract = compute(base_data, tar_data, not_extract.get(tar_sys_lab))
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
        # pros = pro_names[i].split(',')
        cypher = f'match (n:{sys_label}:{ent_lab}) '
        cypher += 'return id(n) as id_, '
        for p in range(len(pro_names)):
            cypher += 'n.' + pro_names[p] + f' as pro_{p}, '
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
        f'as id_,'
    for p in range(len(pros)):
        cypher += 'm.' + pros[p] + f' as pro_{p}, '
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
        return returned, not_extract
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
    # if len(base_pros) > 1:  # 说明出现多类实体处在同一级别上
    #     # 在当前的设置中，这个条件分支永远也不会被执行到
    #     base_data = []
    #     for i in range(len(base_pros)):
    #         base_pro = base_pros[i].split(',')
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
        # if len(tar_pros) > 1:  # 说明出现类多类实体处在同一级别上
        #     tar_data = []
        #     for i in range(len(tar_pros)):
        #         tar_pro = tar_pros[i].split(',')
        #         tar_data.extend(get_data2(tar_sys, tar_pro, level, tar_p_id,
        #                                   not_extract.get(tar_sys), i))
        tar_data = get_data2(tar_sys, tar_pros, level, tar_p_id,
                             not_extract.get(tar_sys))
        if not tar_data:
            continue
        similarities[tar_sys], _not_extract = compute(base_data, tar_data,
                                                      not_extract.get(tar_sys))
        not_extract[tar_sys] = _not_extract
    res = [i for i in similarities if i]
    if not res:
        return no_similarity(base_data, base_sys)
    df = combine_sim(similarities, base_sys)
    label_df = label_df.drop(base_sys, axis=1)
    root_results_bak = parent_ids.copy()
    root_results_bak.pop(systems.index(base_sys))
    return df.append(fuse_in_same_level(label_df, root_results_bak, level, not_extract))


def update_progress(task_id, integer):
    conn = connect(**eval(mysql_res))
    with conn.cursor() as cr:
        cr.execute(f"update gd_fuse set fuse_speed={integer} where id='{task_id}'")
    conn.commit()


def save_detail_to_mysql(node, task_id, new_batch):
    """将明细数据存储至mysql"""
    conn = connect(**eval(mysql_res))
    # 遍历node中的所有节点及其子节点，然后在图数据库中查询对应的属性信息，统计后写入
    # MySQL

    def traverse(node, value_list):
        """遍历node，获取所有的融合值"""
        value_list.append(node.value)
        if node.children:
            for n in node.children:
                value_list = traverse(n, value_list)
        return value_list

    value_list = traverse(node, [])
    for v in value_list:
        info = get_info_from_neo4j(v)
        with conn.cursor() as cr:
            sql = f"""
                insert into gd_fuse_check_result values(
                "{str(uuid1())}", "{task_id}", "{new_batch}",
                "{info[0]}", "{info[1]}", "{info[2]}", "{info[3]}",
                "{info[4]}", "{info[5]}", "{info[6]}", "{info[7]}",
                "{info[8]}", "{info[9]}")
            """
            cr.execute(sql)
    conn.commit()


def get_info_from_neo4j(values: list):
    """根据id，获取相应的属性"""
    graph = Graph(neo4j_url, auth=auth)
    is_exist_dict = {}
    id_node_dict = {}
    id_ = -1
    status = 1
    space = ''
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            is_exist_dict[BASE_SYS_ORDER[i]] = 1
            id_node_dict[BASE_SYS_ORDER[i]] = int(v)
            if id_ == -1:  # 遍历id列表的时候，从第一个不为空的id所对应的节点来获取属性
                id_ = int(v)
                space = BASE_SYS_ORDER[i]
        else:
            is_exist_dict[BASE_SYS_ORDER[i]] = 2
            status = 2
            id_node_dict[BASE_SYS_ORDER[i]] = -1

    # 要从单网架图获取的属性名的列表 todo
    pros = OrderedDict({
        "orgCode": "merit_orgCode",
        "provinceCompany": "merit_province",
        "cityCompany": "merit_city",
        "countyCompany": "merit_district",
        "equipmentType": "entityType",
        "equipmentName": "name"
    })
    cypher = f'match(n:{space}) where id(n)={id_} return n.'
    for p in pros:
        cypher += pros[p]
        cypher += f' as {p},n.'
    cypher = cypher[: -3]
    data = graph.run(cypher).data()[0]
    # 按照顺序来将要存储的值写入列表
    data_list = []
    for p in pros:
        data_list.append(data[p])

    data_list.append(id_)
    data_list.extend([str(is_exist_dict), str(id_node_dict)])
    data_list.append(status)
    return data_list


def update_status(start_time, task_id):
    conn = connect(**eval(mysql_res))
    end_time = strftime("%Y-%m-%d %H:%M:%S")
    with conn.cursor() as cr:
        cr.execute(f"update gd_fuse set fuse_status=5, fuse_speed=100, "
                   f"start_time='{start_time}', end_time='{end_time}' "
                   f"where id='{task_id}'")
    conn.commit()
