# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/28 4:31 下午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""


class Nodes:
    def __init__(self, label, value):
        self.children = {}
        self.label = label
        self.value = value

    def add_child(self, label, value, overwrite=False):
        max_ind = max(self.children) if self.children else 0
        child = Nodes(label, value)
        self.children[max_ind + 1] = child
        if overwrite:
            child.value = value
