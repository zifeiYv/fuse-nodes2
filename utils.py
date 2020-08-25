# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : utils.py                        #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/15                      #
#                                                                   #
#                     Last Update : 2020/08/25                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
import pandas as pd
from py2neo import Graph
from pymysql import connect
from configparser import ConfigParser
import numpy as np
from text_sim_utils import sims

cfg = ConfigParser()
with open('./config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
mysql_cfg = cfg.get('mysql', 'mysql_cfg')
mysql_res = cfg.get('mysql', 'mysql_res')


class Nodes:
    """定义一个类来存储一个单独的子图。

    对于任意一行融合根节点的数据得到的结果，都是一个根节点，从该根节点出发，结合配置文件中
    的融合结构一直融合到最后一级实体，即得到来一颗树，称为子图。

    存储该子图，以便于后续在Neo4j中生成新的融合图。
    """
    def __init__(self, label: str, value: list, rel: str = None):
        """实例化时，需要传入两个参数。

        Args:
            label: 该层级的设备类型的标签
            value: 按照配置文件中的系统的顺序，依次记录该实体对应于各个系统中的节点的id
            rel: 记录上级实体到本实体的关系标签，如果是根节点，那么为None
        """
        self.children = []
        self.label = label
        self.value = value
        self.rel = rel

    def add_child(self, node):
        """此方法的作用是，为融合后的实体添加子实体。

        各个参数的含义与实例化时的含义相同。

        Args:
            node(Nodes):

        Returns:

        """
        self.children.append(node)


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
        for i in range(sim_matrix.shape[0]):
            if i not in x:
                res.append([i, np.nan])
        return res


class CheckError(Exception):
    pass


def sort_sys(label_df) -> dict:
    """根据配置文件的系统与实体标签，计算其中的基准系统，按照顺序进行排列。

    将每一列看成一个二进制数字，非NaN值写成1，NaN值写成0，
    只需要比较二进制数字的大小即可。

    """
    order, res = {}, {}
    for col in label_df.columns:
        _str = ['0' if isinstance(i, float) else '1' for i in label_df[col]]
        order[col] = int("".join(_str), 2)
    sorted_order = sorted(order.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(order)):
        res[i] = sorted_order[i][0]
    return res


def check(task_id):
    """用于检查参数有效性的函数"""
    # 数据库连接是否正常
    for i in (mysql_res, mysql_cfg):
        try:
            connect(**eval(i))
        except Exception as e:
            raise CheckError(e)

    label, pro, trans, _ = get_paras(task_id)  # 从任务id，获取相关参数并处理成DataFrame的格式

    sys_num = label.shape[1]
    sys_labels = label.columns.values
    print(f"共指定了{sys_num}个系统，其标签分别为：{list(sys_labels)}")
    if sys_num < 2:
        raise CheckError("至少需要指定两个系统才能进行融合")

    ent_num = label.shape[0]
    print(f"共指定了{ent_num}种实体")

    for df in ['pro', 'trans']:
        df_ = eval(df)
        if df_.shape[1] != label.shape[1]:
            raise CheckError(f"`{df}`与`label`中的系统数量不符")
        # noinspection PyTypeChecker
        if not all(df_.columns == label.columns):
            raise CheckError(f"`{df}`与`label`中的系统名称不符")

    for s in sys_labels:
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
        cr.execute(f"select label from gd_fuse where id='{task_id}'")
        merged_label = cr.fetchone()[0]
        cr.execute(f"select space_label, ontological_label, ontological_weight, "
                   f"ontological_mapping_column_name from gd_fuse_attribute t where t.fuse_id='"
                   f"{task_id}'")
        info = cr.fetchall()

    # 由于给本体设置的权重不一定是从1开始的连续数字，
    # 因此需要对权重进行特别的处理后进行排序。
    weights = list(set([i[2] for i in info]))
    weights.sort()
    num_rows = len(weights)
    columns = list(set([i[0] for i in info]))
    label = pd.DataFrame(index=range(num_rows), columns=columns)
    pro = pd.DataFrame(index=range(num_rows), columns=columns)
    for c in columns:
        for t in info:
            if t[0] != c:
                continue
            else:
                if isinstance(label.loc[num_rows-weights.index(t[2])-1, c], float):  # 说明该位置上尚未有值
                    label.loc[num_rows-weights.index(t[2])-1, c] = t[1]
                else:  # 否则，将新的值追加到原来的值后面，用英文分号分割
                    # todo: 以分号分割并进行后续处理的功能目前尚未完成，因此在此处进行了限制，
                    #   即只如果某两个实体具有相同的权重，那么只会保留最后一个，至于是哪一个，
                    #   是无法确定的。
                    #   下同。
                    # label.loc[num_rows-weights.index(t[2])-1, c] += label.loc[
                    # num_rows-weights.index(t[2])-1, c] + ';' + t[1]
                    label.loc[num_rows-weights.index(t[2])-1, c] = t[1]

                if isinstance(pro.loc[num_rows-weights.index(t[2])-1, c], float):
                    pro.loc[num_rows-weights.index(t[2])-1, c] = t[3]
                else:
                    # todo：同上
                    # pro.loc[num_rows-weights.index(t[2])-1, c] = pro.loc[num_rows-weights.index(
                    #     t[2])-1, c] + ';' + t[3]
                    pro.loc[num_rows-weights.index(t[2])-1, c] = t[3]

    trans = pro.copy()

    return label, pro, trans, merged_label
