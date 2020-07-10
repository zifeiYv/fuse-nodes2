# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/22 5:40 下午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com

Main functions to create subgraphs basing on fuse results.
"""
from configparser import ConfigParser
from py2neo import Graph, Node, Relationship

cfg = ConfigParser()

with open('config_files/application.cfg') as f:
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))

with open('config_files/neo4j-structure.cfg') as f:
    cfg.read_file(f)
fused_entities = cfg.get('fuse', 'entities').split(',')
fused_rel = cfg.get('fuse', 'relationships').split(',')
fused_pros = cfg.get('fuse', 'properties').split('&')
fused_pros = [i.split(',') for i in fused_pros]

cms_entities = cfg.get('cms', 'entities').split(',')

if len(fused_entities) != len(cms_entities):
    fused_entities = cms_entities


def delete_old(label):
    """Delete ole fuse results.

    Args:
        label(str): Node label

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    graph.run(f"match (n:`{label}`)-[r]-() delete r")
    graph.run(f"match (n:`{label}`) delete n")


def create_subgraph(label, sub_graph):
    """According to `sub_graph`, create a subgraph in neo4j.

    Args:
        label(str): New label from generated nodes
        sub_graph(dict): A dict contains fuse results

    Returns:

    """
    graph = Graph(neo4j_url, auth=auth)
    tx = graph.begin()
    # noinspection PyTypeChecker
    root_node = generate_node([fused_entities[0], label], fused_pros[0], *sub_graph['val'])
    tx.create(root_node)

    def func(parent_node, graph_data, i):
        data = graph_data['children']
        for k in data:
            node = generate_node([fused_entities[i], label], fused_pros[i], *data[k]['val'])
            tx.create(node)
            rel = fused_rel[i-1]
            tx.create(Relationship(parent_node, rel, node))
            if data[k]['children'] == {}:
                continue
            if i != len(fused_entities) - 1:
                j = i + 1
                graph_data = data[k]
                func(node, graph_data, j)

    func(root_node, sub_graph, 1)
    tx.commit()


def generate_node(label, pros, node_id1=None, node_id2=None, node_id3=None):
    """Generate fused node according to ids.

    Args:
        label(list): Label of new node
        pros(list): Properties that transferred from original nodes to new nodes
        node_id1(str): Original node's id
        node_id2(str): Original node's id
        node_id3(str): Original node's id

    Returns:
        A `py2neo.Node` object
    """
    graph = Graph(neo4j_url, auth=auth)
    data = {'cmsId': node_id1 if node_id1 else '',
            'pmsId': node_id2 if node_id2 else '',
            'gisId': node_id3 if node_id3 else ''}

    for p in pros:
        # if node_id1:
        #     val1 = graph.run(f"match(n) where id(n)={node_id1} return n.{p} as p").data()[0]['p']
        # else:
        #     val1 = ''
        # if node_id2:
        #     val2 = graph.run(f"match(n) where id(n)={node_id2} return n.{p} as p").data()[0]['p']
        # else:
        #     val2 = ''
        # if node_id3:
        #     val3 = graph.run(f"match(n) where id(n)={node_id3} return n.{p} as p").data()[0]['p']
        # else:
        #     val3 = ''
        # if val1 is not None:  # means `p` does in the database
        #     data[f'cms-{p}'] = val1
        # if val2 is not None:
        #     data[f'pms-{p}'] = val2
        # if val3 is not None:
        #     data[f'gis-{p}'] = val3
        # Match `p` in 'cms'/'pms'/'gis' order and break when getting one valid value
        val = ''
        for id_ in [node_id1, node_id2, node_id3]:
            if id_ is not None:
                val = graph.run(f"match(n) where id(n)={id_} return n.{p} as p").data()[0]['p']
                if val:
                    break
                else:
                    continue
        data[f'{p}'] = val

    return Node(*label, **data)


if __name__ == '__main__':
    delete_old('merge')
