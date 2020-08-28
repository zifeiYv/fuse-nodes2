u"""
@Time   : 2020/6/23 3:18 \u4e0b\u5348
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com

Functions used for distributed fuse.
"""
from py2neo import Graph
from configparser import ConfigParser
import numpy as np
from text_sim_utils import sims
from subgraphs import create_subgraph
from gen_logger import gen_logger
import pymysql, datetime
logger = gen_logger('fuse.log', 1)
logger.info('Get necessary information before starting fusing...')
cfg = ConfigParser()
with open('./config_files/application.cfg') as (f):
    cfg.read_file(f)
neo4j_url = cfg.get('neo4j', 'url')
auth = eval(cfg.get('neo4j', 'auth'))
mysql = cfg.get('mysql', 'mysql')
THRESHOLD = cfg.getfloat('threshold', 'threshold')
with open('./config_files/neo4j-structure.cfg') as (f):
    cfg.read_file(f)
cms, pms, gis = cfg.get('system_label', 'cms'), cfg.get('system_label', 'pms'), cfg.get('system_label', 'gis')
fused_entities = cfg.get('fuse', 'entities').split(',')
if list(map(len, (cms, pms, gis))).count(0) == 1:
    if not gis:
        flag = 'cp'
        logger.info('Fusing will start between cms and pms')
    else:
        if not pms:
            flag = 'cg'
            logger.info('Fusing will start between cms and gis')
        else:
            flag = 'pg'
            logger.info('Fusing will start between pms and gis')
else:
    flag = 'all'
    logger.info('Fusing will start among 3 systems')
cms_entities = cfg.get('cms', 'entities').split(',')
cms_rel = cfg.get('cms', 'relationships').split(',')
cms_pros = cfg.get('cms', 'properties').split('&')
pms_entities = cfg.get('pms', 'entities').split(',')
pms_rel = cfg.get('pms', 'relationships').split(',')
pms_pros = cfg.get('pms', 'properties').split('&')
gis_entities = cfg.get('gis', 'entities').split(',')
gis_rel = cfg.get('gis', 'relationships').split(',')
gis_pros = cfg.get('gis', 'properties').split('&')
weight = cfg.get('weight', 'weight').split('&')
logger.info('Done')

def fuse_and_create(args):
    """Used for multi-processes mode"""
    label, res, cur, tot = args
    logger.info(f'''Progress:{(cur + 1)}/{tot}''')
    children = fuse_other_nodes(1, res)
    sub_graph = {'val':res,  'children':children}
    logger.info('  Saving fuse results to mysql...')
    save_to_mysql(sub_graph)
    logger.info('  Done')
    logger.info('  Creating subgraphs...')
    create_subgraph(label, sub_graph)
    logger.info('  Done')


def fuse_root_nodes(is_multi=False, processes=0):
    """Function to fuse the first label in config files"""
    ce, pe, ge = cms_entities[0], pms_entities[0], gis_entities[0]
    cp, pp, gp = get_property(0)
    w = eval(weight[0])
    logger.info('Getting root node data...')
    cms_data = get_root_data(cms, ce, cp, w)
    pms_data = get_root_data(pms, pe, pp, w)
    gis_data = get_root_data(gis, ge, gp, w)
    logger.info('Done')
    res = validate_data_and_fuse(0, cms_data, pms_data, gis_data)
    logger.info('Root data fuse finish')
    if is_multi:
        logger.info(f'''Another {processes} are running background to fuse other nodes and generate subgraphs
''')
    else:
        logger.info('Fuse other nodes and generate subgraphs in current process\n')
    return res


def fuse_other_nodes(start_index, root_result):
    """Recursively fuse all children and grand children basing on `root_result`
    
    Args:
        start_index(int): Which label to start with
        root_result(list): A list contains root node ids
    
    Returns:
    
    """
    graph = Graph(neo4j_url, auth=auth)
    result_dict = {}
    i = start_index
    ce, pe, ge = cms_entities[i], pms_entities[i], gis_entities[i]
    cp, pp, gp = get_property(i)
    w = eval(weight[i])
    if root_result.count(None) == 0:
        cms_data = get_data(graph, root_result[0], cms, 'cms', ce, cp, w, i)
        pms_data = get_data(graph, root_result[1], pms, 'pms', ce, cp, w, i)
        gis_data = get_data(graph, root_result[2], gis, 'gis', ce, cp, w, i)
    else:
        if root_result.count(None) == 1:
            if root_result[0] is None:
                cms_data = []
                pms_data = get_data(graph, root_result[1], pms, 'pms', ce, cp, w, i)
                gis_data = get_data(graph, root_result[2], gis, 'gis', ce, cp, w, i)
            else:
                if root_result[1] is None:
                    pms_data = []
                    cms_data = get_data(graph, root_result[0], cms, 'cms', ce, cp, w, i)
                    gis_data = get_data(graph, root_result[2], gis, 'gis', ce, cp, w, i)
                else:
                    gis_data = []
                    cms_data = get_data(graph, root_result[0], cms, 'cms', ce, cp, w, i)
                    pms_data = get_data(graph, root_result[1], pms, 'pms', ce, cp, w, i)
        else:
            if root_result[0] is not None:
                pms_data, gis_data = [], []
                cms_data = get_data(graph, root_result[0], cms, 'cms', ce, cp, w, i)
            else:
                if root_result[1] is not None:
                    cms_data, gis_data = [], []
                    pms_data = get_data(graph, root_result[1], pms, 'pms', ce, cp, w, i)
                else:
                    cms_data, pms_data = [], []
                    gis_data = get_data(graph, root_result[2], gis, 'gis', ce, cp, w, i)
                res = validate_data_and_fuse(i, cms_data, pms_data, gis_data)
                if res is None:
                    return {}
                if start_index < len(cms_entities) - 1:
                    start_index += 1
                    for j in range(len(res)):
                        result_dict[j] = {'val':res[j],  'children':fuse_other_nodes(start_index, res[j])}

                else:
                    for j in range(len(res)):
                        result_dict[j] = {'val':res[j], 
                         'children':{}}

                return result_dict


def get_property(i):
    """Get all properties that are used to calculate similarity from `./config_files/neo4j-structure.cfg`. In order
    to distinguish TEXT property and ENUM property, using a nested list to store them.
    
    Args:
        i(int): Number of entity labels
    
    Returns: Tuple of lists of lists
    
    """
    cms_ = eval(cms_pros[i])
    pms_ = eval(pms_pros[i])
    gis_ = eval(gis_pros[i])
    cms_list = [
     cms_['TEXT'], cms_['ENUM']]
    pms_list = [pms_['TEXT'], pms_['ENUM']]
    gis_list = [gis_['TEXT'], gis_['ENUM']]
    return (
     cms_list, pms_list, gis_list)


def get_root_data(sys, ent, prop, w):
    """Get root data a.k.a. the first entity to be fused, from neo4j.
    
    Only properties in `p` as well as ``id``, which is the Universally Unique Identifier(UUID),  will be extracted. In
    the later calculation, ``id`` will be used to associate to original node(s).
    
    For TEXT properties, add `_TEXT` to the end of original property name as its alias; for ENUM properties,
    add `_ENUM` to the end of original property name as its alias.
    
    Args:
        sys(str): System label
        ent(str): Entity label
        prop(list): List of properties, a nested object
        w(dict): Weight dict
    
    Returns:
        List of dicts, each dict stands for a node in neo4j
    
    """
    graph = Graph(neo4j_url, auth=auth)
    count = 1
    cypher = f'''match (n:{sys}:{ent}) return id(n) as id_, n.'''
    for i in range(len(prop[0])):
        cypher += prop[0][i] + f''' as `TEXT_{count}_{(w['TEXT'][i])}`, '''
        count += 1

    for i in range(len(prop[1])):
        cypher += prop[1][i] + f''' as `ENUM_{count}_{(w['ENUM'][i])}`, '''
        count += 1

    cypher = cypher[:-2]
    data = graph.run(cypher).data()
    return data


def get_data(graph_, father, sys, sys_type, ent, prop, w, i):
    """Get data from neo4j. The extraction is the same with :func:`get_root_data`.
    
    Args:
        graph_: A `neo4j.Graph` object
        father(str): Father node id
        sys(str): System label
        sys_type(str): System type, one of 'cms', 'pms' and 'gis'
        ent(str): Entity label
        prop(list): List of properties, a nested object
        w(dict): Weight dict
        i(int): Relationship's number
    
    Returns:
        List of dicts, each dict stands for a node in neo4j
    """
    if sys_type == 'cms':
        rel = cms_rel[i - 1][1:][:-1]
    else:
        if sys_type == 'pms':
            rel = pms_rel[i - 1][1:][:-1]
        else:
            rel = gis_rel[i - 1][1:][:-1]
        count = 1
        cypher = f'''match(m){rel}(n:{sys}:{ent}) where id(m)={father} return id(n) as id_, n.'''
        for i in range(len(prop[0])):
            cypher += prop[0][i] + f''' as `TEXT_{count}_{(w['TEXT'][i])}`, '''
            count += 1

        for i in range(len(prop[1])):
            cypher += prop[1][i] + f''' as `ENUM_{count}_{(w['ENUM'][i])}`, '''
            count += 1

        cypher = cypher[:-2]
        data = graph_.run(cypher).data()
        return data


def validate_data_and_fuse(i, cms_data, pms_data, gis_data):
    """Make arguments for `compute_sim_and_combine` valid and then call it.
    
    Args:
        i(int): Fused nodes's number, used to generate pretty log
        cms_data(list): List of dicts
        pms_data(list): List of dicts
        gis_data(list): List of dicts
    
    Returns:
    
    """
    logger.info(f'''{('  ' * i)}Getting data...''')
    if not cms_data:
        logger.warning(f'''{('  ' * i)}No data from cms system!''')
        cms_data = None
    if not pms_data:
        logger.warning(f'''{('  ' * i)}No data from pms system!''')
        pms_data = None
    if not gis_data:
        logger.warning(f'''{('  ' * i)}No data from gis system!''')
        gis_data = None
    none_counts = sum(map(lambda x: x is None, (cms_data, pms_data, gis_data)))
    if none_counts == 3:
        logger.warning(f'''{('  ' * i)}Got nothing from all systems''')
        return
    if none_counts == 2:
        logger.warning(f'''{('  ' * i)}Got data from one system, just copy data''')
    else:
        if none_counts == 1:
            logger.warning(f'''{('  ' * i)}Got data from two systems''')
        else:
            logger.info(f'''{('  ' * i)}Got from three systems''')
        logger.info(f'''{('  ' * i)}Done''')
        return compute_sim_and_combine(i, none_counts, cms_data, pms_data, gis_data)


def compute_sim_and_combine(i, none_counts, cms_data=None, pms_data=None, gis_data=None):
    """For the given data of three systems, compute similarity matrix and get the most similar pairs,
    then combing node ids together into a nested list.
    
    Note that at most one of the three arguments could be `None`, which means this function can always computing
    similarities between/among different systems.
    
    Arguments could be empty list, which means didn't get valid data from neo4j, and may cause errors.
    So it is necessary to judge whether parameters are legal before calling.
    
    Args:
        i(int): Fused nodes's number, used to generate pretty log
        none_counts(int): 0, 1 or 2, `None` values' number
        cms_data(list[dict]): Node data from cms
        pms_data(list[dict]): Node data from pms
        gis_data(list[dict]): Node data from gis
    
    Returns:
    
    """
    computer = Computation(THRESHOLD)
    if none_counts == 1:
        if not cms_data:
            logger.warning(f'''{('  ' * i)}Similarities will be computed in only two systems(pms, gis)''')
            res = computer.compute(pms_data, gis_data)
            logger.info(f'''{('  ' * i)}Done''')
            if res is None:
                logger.warning(f'''{('  ' * i)}Totally different!''')
                return res
            logger.info(f'''{('  ' * i)}Combining node ids...''')
            for i_ in res:
                i_.insert(0, None)
                if i_[1] is not None:
                    i_[1] = pms_data[i_[1]]['id_']
                if i_[2] is not None:
                    i_[2] = gis_data[i_[2]]['id_']

            logger.info(f'''{('  ' * i)}Done''')
            return res
        if not pms_data:
            logger.warning(f'''{('  ' * i)}Similarities will be computed in only two systems(cms, gis)''')
            res = computer.compute(cms_data, gis_data)
            logger.info(f'''{('  ' * i)}Done''')
            if res is None:
                logger.warning(f'''{('  ' * i)}Totally different!''')
                return res
            logger.info(f'''{('  ' * i)}Combining node ids...''')
            for i_ in res:
                i_.insert(1, None)
                if i_[0] is not None:
                    i_[0] = cms_data[i_[0]]['id_']
                if i_[2] is not None:
                    i_[2] = gis_data[i_[2]]['id_']

            logger.info(f'''{('  ' * i)}Done''')
            return res
        logger.warning(f'''{('  ' * i)}Similarities will be computed in only two systems(cms, pms)''')
        res = computer.compute(cms_data, pms_data)
        logger.info(f'''{('  ' * i)}Done''')
        if res is None:
            logger.warning(f'''{('  ' * i)}Totally different!''')
            return res
        logger.info(f'''{('  ' * i)}Combining node ids...''')
        for i_ in res:
            i_.insert(2, None)
            if i_[0] is not None:
                i_[0] = cms_data[i_[0]]['id_']
            if i_[1] is not None:
                i_[1] = pms_data[i_[1]]['id_']

        logger.info('Done')
        return res
    else:
        if none_counts == 2:
            res = []
            if cms_data is not None:
                for i in cms_data:
                    res.append([i['id_'], None, None])

            if pms_data is not None:
                for i in pms_data:
                    res.append([None, i['id_'], None])

            if gis_data is not None:
                for i in gis_data:
                    res.append([None, None, i['id_']])

            return res
    logger.info(f'''{('  ' * i)}Computing similarities...''')
    res1 = computer.compute(cms_data, pms_data)
    res2 = computer.compute(cms_data, gis_data)
    res3 = computer.compute(pms_data, gis_data)
    logger.info(f'''{('  ' * i)}Done''')
    logger.info(f'''{('  ' * i)}Combining node ids...''')
    res = []
    for j in res1:
        if j.count(None) == 0:
            cms_id, pms_id = cms_data[j[0]]['id_'], pms_data[j[1]]['id_']
            for k in res2:
                if k[0] is not None:
                    if cms_id == cms_data[k[0]]['id_']:
                        gis_id = gis_data[k[1]]['id_'] if k[1] is not None else None
                        res.append([cms_id, pms_id, gis_id])

        else:
            if j[0] is None:
                pms_id = pms_data[j[1]]['id_']
                for k in res3:
                    if k[0] is not None:
                        if pms_id == pms_data[k[0]]['id_']:
                            gis_id = gis_data[k[1]]['id_'] if k[1] is not None else None
                            res.append([None, pms_id, gis_id])

            else:
                cms_id = cms_data[j[0]]['id_']
                for k in res2:
                    if k[0] is not None:
                        if cms_id == cms_data[k[0]]['id_']:
                            gis_id = gis_data[k[1]]['id_'] if k[1] is not None else None
                            res.append([cms_id, None, gis_id])

    logger.info(f'''{('  ' * i)}Done''')
    return res


def save_to_mysql(sub_graph):
    """Save fuse results to MySQL"""
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = pymysql.connect(**eval(mysql))
    g = Graph(neo4j_url, auth=auth)
    sql = f'''insert into fuse_results(period, cms_id, pms_id, gis_id, city_code, county_code, gds_code, sys_type, equip_type) values ('{now}', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')'''

    def func(i, graph):
        if i == 0:
            args = combine_args(graph['val'])
            args.append(fused_entities[i])
            with conn.cursor() as (cr):
                cr.execute(sql % tuple(args))
        else:
            for j in graph:
                args = combine_args(graph[j]['val'])
                args.append(fused_entities[i])
                with conn.cursor() as (cr):
                    cr.execute(sql % tuple(args))
                child = graph[j]['children']
                if child is not {}:
                    func(i + 1, child)

    def combine_args(nodes):
        args = []
        _check = []
        city_code, county_code, gds_code = ('', '', '')
        for i in range(len(nodes)):
            if nodes[i] is not None:
                _check.append(1)
                data = g.run(f'''match(n) where id(n)={(nodes[i])} return n.cityCode as city_code, n.mRID as mrid, n.countyCode as county_code, n.gdsCode as gds_code''').data()
                if data:
                    city_code = data[0]['city_code'] if data[0]['city_code'] else ''
                    county_code = data[0]['county_code'] if data[0]['county_code'] else ''
                    gds_code = data[0]['gds_code'] if data[0]['gds_code'] else ''
                    args.append(data[0]['mrid'])
                else:
                    _check.append(0)
                    args.append('')

        args.extend([city_code, county_code, gds_code])
        if _check == [1, 1, 1]:
            args.append('yx-pms-gis')
        if _check == [1, 0, 0]:
            args.append('yx')
        if _check == [1, 0, 1]:
            args.append('yx-gis')
        if _check == [1, 1, 0]:
            args.append('yx-pms')
        if _check == [0, 1, 1]:
            args.append('pms-gis')
        if _check == [0, 1, 0]:
            args.append('pms')
        if _check == [0, 0, 1]:
            args.append('gis')
        return args

    func(0, sub_graph)
    func(1, sub_graph['children'])
    conn.commit()


class Computation:
    """A class to calculate similarity between two lists of neo4j nodes"""

    def __init__(self, thresh=0.75):
        """Pass a threshold value when instantiation, default 0.75"""
        self.thresh = thresh

    def compute(self, data1, data2):
        """Main function of computation.
        
        Args:
            data1(list[dict]): List of dicts
            data2(list[dict]): List of dicts
        
        Returns:
            None or result list.
        
        """
        sim = np.zeros(shape=(len(data1), len(data2)))
        for i in range(len(data1)):
            for j in range(len(data2)):
                sim[(i, j)] = self._Computation__compute(data1[i], data2[j])

        return self._Computation__matching(sim)

    @staticmethod
    def __compute(dict1, dict2):
        """Extract data in `dict1` and `dict2` and compute the similarity
        
        Args:
            dict1(dict): A neo4j node
            dict2(dict): A neo4j node
        
        Returns:
            Similarity value
        """
        sim = 0
        for k in dict1:
            if k == 'id_':
                continue
            else:
                w = float(k.split('_')[-1])
                if k.startswith('TEXT'):
                    sim += w * sims(dict1[k], dict2[k])
                elif dict1[k] == dict2[k]:
                    sim += w
                    continue

        return sim

    def __match(self, sim_matrix):
        """Calculate the most similar entity pairs according to `sim_matrix`.
        
        Note that `numpy.array.argmax()` only return the first index of maximum value even if there are more than
        one maximums, which may bring some confusion in certain cases.
        
        Args:
            sim_matrix: similarity matrix
        
        Returns: List of lists
        
        """
        if np.sum(sim_matrix) == 0:
            return
        res = []
        arg0 = sim_matrix.argmax(axis=0)
        arg1 = sim_matrix.argmax(axis=1)
        for i in range(len(arg0)):
            if sim_matrix[(arg0[i], i)] < self.thresh:
                sim_matrix[:, i] = 0
                if [None, i] not in res:
                    res.append([None, i])
                else:
                    if arg1[arg0[i]] == i:
                        res.append([arg0[i], i])
                        sim_matrix[:, i] = 0
                        sim_matrix[arg0[i]] = 0

        r = self._Computation__match(sim_matrix)
        if r:
            return res + r
        else:
            return res

    def __matching(self, sim_matrix):
        """After `self.__match`, some indexes across axis 0 and/or axis 1 may be left, so make them match with `None`.
        
        Args:
            sim_matrix: similarity matrix
        
        Returns: List of lists
        
        """
        res = self._Computation__match(sim_matrix)
        if res is None:
            return
        else:
            x = set([i[0] for i in res])
            y = set([i[1] for i in res])
            for i in range(sim_matrix.shape[0]):
                if i not in x:
                    res.append([i, None])

            for j in range(sim_matrix.shape[1]):
                if j not in y:
                    res.append([None, j])

            return res