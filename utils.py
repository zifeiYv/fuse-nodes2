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
#                     Last Update : 2020/07/27                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
from collections import OrderedDict


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
