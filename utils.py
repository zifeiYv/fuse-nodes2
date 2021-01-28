# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
from text_sim_utils import sims


class Nodes:
    """A class to store subgraphs.

    Each row of the output of :py:func:`~fuse.fuse_root_nodes` contains all original ids
    for a new node that needed to be generated, which is called the start node.

    Along with this start node, we can find all of their children and fuse them
    together. The fuse results are stored as a instance of this class.
    """
    def __init__(self, label: str, value: list, rel: str = None):
        """

        Args:
            label: Entities' label
            value: A list of original ids, whose order is defined in configuration file
            rel: Relation name of parent node to this node, `None` for root node
        """
        self.children = []
        self.label = label
        self.value = value
        self.rel = rel

    def add_child(self, node):
        """Add child node to an instance.
        """
        self.children.append(node)


class Computation:
    """
    传入两个由字典组成的列表，每个字典代表一个实体对象， ``compute`` 方法计算两个列表中对象的相似度，
    最后返回一个嵌套的列表，最内层列表中的数值为每个字典在传入列表中的索引。"""

    def __init__(self, thresh=0.75):
        """判定为同一对象的阈值，默认为0.75，可以通过配置文件修改"""
        self.thresh = thresh
        self.match_res = []
        self.x = []
        self.y = []

    def compute(self, base_data, tar_data):
        """对外提供的计算相似度的方法接口。

        Args:
            base_data(list[dict]): 字典组成的列表，每个字典都表示一个节点
            tar_data(list[dict]): 字典组成的列表，每个字典都表示一个节点

        Returns:
            None/List

        """
        if self.match_res:
            self.match_res = []
        if self.x:
            self.x = []
        if self.y:
            self.y = []
        sim = np.zeros(shape=(len(base_data), len(tar_data)), dtype=np.float16)
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
        if np.sum(sim_matrix, dtype=np.float64) == 0:
            return

        arg0 = sim_matrix.argmax(axis=0)                 # 每一列的最大值的位置
        arg1 = sim_matrix.argmax(axis=1)                 # 每一行的最大值的位置
        for i in range(len(arg0)):                       # 按列遍历
            if i in self.y or arg0[i] in self.x:         # 已经计算
                continue                                 # 跳过
            if sim_matrix[(arg0[i], i)] < self.thresh:   # i列的最大值小于阈值
                sim_matrix[:, i] = 0                     # 整列设置为0
                # if [None, i] not in self.match_res:      # 如果该匹配对不在结果列表中
                #     self.match_res.append([None, i])     # 追加匹配对到结果列表中
                #     self.y.append(i)                     # 追加到已计算的y值列表
                continue
            else:                                        # i列的最大值大于等于阈值
                if arg1[arg0[i]] == i:                   # 第i列的最大值刚好也是该值所处行的最大值
                    self.match_res.append([arg0[i], i])  # 将该匹配对追加到结果列表中
                    sim_matrix[:, i] = 0                 # 整列设置为0
                    sim_matrix[arg0[i]] = 0              # 整行设置为0
                    self.x.append(arg0[i])               # 追加到已计算的x值列表
                    self.y.append(i)                     # 追加到已计算的y值列表
        self.__match(sim_matrix)

    def __matching(self, sim_matrix):
        """在调用`self.__match`之后，在横向或者纵向上可能遗留下一些非0值，对这些值
        进行处理。

        只处理基准表对应的数据，以保证基准表中所有的节点都被计算完成。

        Args:
            sim_matrix: 相似度矩阵

        Returns:
            嵌套的列表

        """
        if sum(sim_matrix.shape) == 2:  # 说明两个系统各只有一个实体进行融合
            if sim_matrix[0, 0] < self.thresh:
                self.match_res.append([0, None])
                self.match_res.append([None, 0])
                return self.match_res

        self.__match(sim_matrix)

        if not self.match_res:
            return np.nan

        for i in range(sim_matrix.shape[0]):
            if i not in self.x:
                self.match_res.append([i, None])

        for j in range(sim_matrix.shape[1]):
            if j not in self.y:
                self.match_res.append([None, j])
        return self.match_res


def sort_sys(label_df) -> dict:
    """根据配置文件的系统与实体标签，计算其中的基准系统，按照顺序进行排列。

    判断方法：将每一列看成一个二进制数字，非NaN值写成1，NaN值写成0，
    只需要比较二进制数字的大小即可。数值越大，则越优先被选为基准系统。

    如果出现数值相等的情况，则按照从左到右的顺序进行判定。

    """
    order, res = OrderedDict(), {}
    for col in label_df.columns:
        str_list = ['0' if isinstance(i, float) else '1' for i in label_df[col]]
        order[col] = int("".join(str_list), 2)  # 拼接到一起并转化为十进制数
    sorted_order = sorted(order.items(), key=lambda x: x[1], reverse=True)  # 按大小排序
    for i in range(len(order)):
        res[i] = sorted_order[i][0]
    return res
