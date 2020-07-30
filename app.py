# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : app.py                          #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/16                      #
#                                                                   #
#                     Last Update : 2020/07/27                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
from self_check import check
from utils import Nodes
from flask import Flask, jsonify
from concurrent.futures import ProcessPoolExecutor
from progressbar import ProgressBar

FUSE_AND_CREATE = True  # 融合完成一个子图，便在Neo4j中创建一个子图

bar = ProgressBar('sub_graph')
bar.create()

url = '/entity_fuse/'
executor = ProcessPoolExecutor(1)

app = Flask(__name__)


@app.route(url)
def func():
    if bar.get() not in (1.0, 0.0):
        return jsonify({'state': 0, "msg": "当前有正在执行的任务，请等待其完成后重试"})
    print("必要的检查...")
    try:
        check()
    except Exception as e:
        print('ERROR:', e)
        return jsonify({'state': 0, 'msg': "配置文件非法，查看控制台输出"})
    print("一切正常")
    executor.submit(main_fuse)
    return jsonify({"state": 1, "msg": "正在后台进行融合任务"})


@app.route(url + 'query_progress/')
def query_progress():
    return jsonify({'state': 1, 'msg': round(bar.get(), 2)})


@app.route(url + 'initialize/')
def initialize():
    """未避免因强制终止程序使得redis中的进度不为1而导致无法进行融合，请求本服务以将其置为0"""
    bar.create()
    return jsonify(({'state': 1, 'msg': '初始化状态成功'}))


def main_fuse():
    from fuse import fuse_root_nodes, fuse_other_nodes, BASE_SYS_ORDER, LABEL, \
        create_node_and_rel, delete_old
    print("开始融合")
    print("删除旧的融合结果...")
    delete_old('merge')
    print("删除完成")

    root_res_df = fuse_root_nodes()
    if root_res_df is None:
        print("根节点融合后无结果，无法继续执行")
    else:
        print("根节点融合完成，开始融合子图")
        base_ent_lab = LABEL[BASE_SYS_ORDER[0]].iloc[0]
        for i in range(len(root_res_df)):
            bar.set((i + 1)/len(root_res_df))
            node = Nodes(base_ent_lab, root_res_df.iloc[i].to_list())
            fuse_other_nodes(1, node, BASE_SYS_ORDER)  # 执行之后，node包含了创建一个子图所需要的完整信息
            if FUSE_AND_CREATE:
                create_node_and_rel(node)
        print("创建新图完成")


if __name__ == '__main__':
    # 生成环境下启动使用`gunicorn -c gunicorn_config.py app:app`
    app.run('0.0.0.0')
