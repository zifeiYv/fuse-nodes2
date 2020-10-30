# -*- coding: utf-8 -*-
"""
File Name  : self_test
Author     : Jiawei Sun
Email      : j.w.sun1992@gmail.com
Start Date : 2020/08/28
Last Update:
"""
from configparser import ConfigParser
from py2neo import Graph


class NotEnoughLabels(Exception):
    pass


class LengthNotMatchError(Exception):
    pass


class NotEnoughProperties(Exception):
    pass


class PropertiesNotMatch(Exception):
    pass


def func(list1, list2, list3=None):
    """If specify `list3`, to check if the number of entity labels of three system are the same;
    else, to check if the number of relationships matches the number of labels.
    """
    if not list3:
        if len(list1) != len(list2) + 1:
            raise LengthNotMatchError("Length of %s and %s not match" % (list1, list2))
    else:
        if not len(list1) == len(list2) == len(list3):
            raise LengthNotMatchError(
                "Length of %s, %s and %s not match" % (list1, list2, list3))


def check():
    """Used before starting the service"""
    cfg = ConfigParser()
    # Is there a `neo4j-structure.cfg` file?
    try:
        with open('./config_files/neo4j-structure.cfg', encoding='utf-8') as f:
            cfg.read_file(f)
    except FileNotFoundError:
        raise

    cms, pms, gis = cfg.get('system_label', 'cms'), \
                    cfg.get('system_label', 'pms'), \
                    cfg.get('system_label', 'gis')
    # Are there two systems at least?
    if list(map(len, (cms, pms, gis))).count(0) > 1:
        raise NotEnoughLabels("Two system labels are needed at least")

    cms_entities = cfg.get('cms', 'entities').split(',')
    cms_rel = cfg.get('cms', 'relationships').split(',')
    cms_pros = cfg.get('cms', 'properties').split('&')
    pms_entities = cfg.get('pms', 'entities').split(',')
    pms_rel = cfg.get('pms', 'relationships').split(',')
    pms_pros = cfg.get('pms', 'properties').split('&')
    gis_entities = cfg.get('gis', 'entities').split(',')
    gis_rel = cfg.get('gis', 'relationships').split(',')
    gis_pros = cfg.get('gis', 'properties').split('&')

    fuse_entities = cfg.get('fuse', 'entities').split(',')
    fuse_rel = cfg.get('fuse', 'relationships').split(',')

    # Does the number of entities, relationships and properties match each other?
    if cms:
        func(cms_entities, cms_rel)
        if len(cms_entities) != len(cms_pros):
            raise LengthNotMatchError(
                "In cms, number of entity and property doesn't match")
    if pms:
        func(pms_entities, pms_rel)
        if len(pms_entities) != len(pms_pros):
            raise LengthNotMatchError(
                "In pms, number of entity and property doesn't match")
    if gis:
        func(gis_entities, gis_rel)
        if len(gis_entities) != len(gis_pros):
            raise LengthNotMatchError(
                "In gis, number of entity and property doesn't match")

    func(fuse_entities, fuse_rel)

    # Are the numbers of entities to be fused of each system the same?
    if list(map(len, (cms, pms, gis))).count(0) == 0:
        func(cms_entities, pms_entities, gis_entities)

    # Does each entity of all system has one property to be calculated at least?
    def f(dict_):
        return dict_['TEXT'] + dict_['ENUM']

    if cms:
        for i in range(len(cms_entities)):
            if not f(eval(cms_pros[i])):
                raise NotEnoughProperties(
                    f"At least one property needed for {cms_entities[i]} of cms, found zero")
    if pms:
        for i in range(len(pms_entities)):
            if not f(eval(pms_pros[i])):
                raise NotEnoughProperties(
                    f"At least one property needed for {pms_entities[i]} of pms, found zero")
    if gis:
        for i in range(len(gis_entities)):
            if not f(eval(gis_pros[i])):
                raise NotEnoughProperties(
                    f"At least one property needed for {gis_entities[i]} of gis, found zero")

    # Are properties' detail of each entity from each system the same?
    if list(map(len, (cms, pms, gis))).count(0) == 0:
        for i in range(len(cms_entities)):
            if not len(eval(cms_pros[i])['TEXT']) == len(eval(pms_pros[i])['TEXT']) == \
                    len(eval(gis_pros[i])['TEXT']):
                raise PropertiesNotMatch(
                    f"'TEXT' properties doesn't match among three systems(property no. {i})")
            if not len(eval(cms_pros[i])['ENUM']) == len(eval(pms_pros[i])['ENUM']) == \
                    len(eval(gis_pros[i])['ENUM']):
                raise PropertiesNotMatch(
                    f"'ENUM' properties doesn't match among three systems(property no. {i})")
    else:
        if not gis:
            for i in range(len(cms_entities)):
                if len(eval(cms_pros[i])['TEXT']) != len(eval(pms_pros[i])['TEXT']):
                    raise PropertiesNotMatch(
                        f"'TEXT' properties doesn't match among three systems(property no. {i})")
                if len(eval(cms_pros[i])['ENUM']) != len(eval(pms_pros[i])['ENUM']):
                    raise PropertiesNotMatch(
                        f"'ENUM' properties doesn't match among three systems(property no. {i})")
        if not pms:
            for i in range(len(cms_entities)):
                if len(eval(cms_pros[i])['TEXT']) != len(eval(gis_pros[i])['TEXT']):
                    raise PropertiesNotMatch(
                        f"'TEXT' properties doesn't match among three systems(property no. {i})")
                if len(eval(cms_pros[i])['ENUM']) != len(eval(gis_pros[i])['ENUM']):
                    raise PropertiesNotMatch(
                        f"'ENUM' properties doesn't match among three systems(property no. {i})")
        if not cms:
            for i in range(len(pms_entities)):
                if len(eval(pms_pros[i])['TEXT']) != len(eval(gis_pros[i])['TEXT']):
                    raise PropertiesNotMatch(
                        f"'TEXT' properties doesn't match among three systems(property no. {i})")
                if len(eval(pms_pros[i])['ENUM']) != len(eval(gis_pros[i])['ENUM']):
                    raise PropertiesNotMatch(
                        f"'ENUM' properties doesn't match among three systems(property no. {i})")

    try:
        with open('config_files/application.cfg', encoding='utf-8') as f:
            cfg.read_file(f)
    except FileNotFoundError:
        raise
    neo4j_url = cfg.get('neo4j', 'url')
    auth = eval(cfg.get('neo4j', 'auth'))
    processes = cfg.getint('distributed', 'processes')
    try:
        graph = Graph(neo4j_url, auth=auth)
        graph.run('return "OK"')
    except Exception:
        raise
    return processes


if __name__ == '__main__':
    check()
