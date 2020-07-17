# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : utils.py                         #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/15                      #
#                                                                   #
#                     Last Update : 2020/07/17                      #
#                                                                   #
#-------------------------------------------------------------------#
"""


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

    目前，这种方式是考虑不周的。如果出现以下情况：

              | sys1 | sys2 | sys3 |
        ------|------|------|------|
        level1|      | Ent  |      |
        ------|------|------|------|
        level2| Ent  | Ent  | Ent  |
        ------|------|------|------|
        level3| Ent  |      |      |
        ------|------|------|------|
        level4| Ent  |      |      |
        ------|------|------|------|
    那么，当前的方法选择的基准系统顺序为：[sys1, sys2, sys3]。在这个基础上，
    因为指定了实体标签的行索引值为0，所以在进行根节点融合时会因选择了NaN值而
    出错。

    对于上述情况，正确的识别顺序应该是：[sys2, sys1, sys3]。

    commit: 已解决上述问题，将每一列看成一个二进制数字，非NaN值写成1，NaN值写成0，
    只需要比较二进制数字的大小即可。

    """
    order, res = {}, {}
    for col in label_df.columns:
        _str = ['0' if isinstance(i, float) else '1' for i in label_df[col]]
        order[int("".join(_str), 2)] = col
    for i in range(len(order)):
        res[i] = order[sorted(order, reverse=True)[i]]
    return res
