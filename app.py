# -*- coding: utf-8 -*-
"""
===============
main entrance
===============
"""
from utils import check
from flask import Flask, jsonify, request
from concurrent.futures import ProcessPoolExecutor
from progressbar import ProgressBar
from fuse import main_fuse
import sys

FUSE_AND_CREATE = True  # 融合完成一个子图，便在Neo4j中创建一个子图

bar = ProgressBar('sub_graph')
bar.create()

url = '/entity_fuse/'
executor = ProcessPoolExecutor(1)

app = Flask(__name__)


@app.route(url, methods=['POST'])
def func():
    if bar.get() not in (1.0, 0.0):
        return jsonify({'state': 0, "msg": "当前有正在执行的任务，请等待其完成后重试"})
    print("必要的检查...")
    task_id = request.json['task_id']
    assert task_id is not None, '必须传入任务id'
    try:
        check(task_id)
    except Exception as e:
        print(__file__, sys._getframe().f_lineno, 'ERROR:', e)
        return jsonify({'state': 0, 'msg': "配置文件非法，查看控制台输出"})
    print("一切正常")
    # main_fuse(task_id)
    executor.submit(main_fuse, task_id)
    return jsonify({"state": 1, "msg": "正在后台进行融合任务"})


@app.route(url + 'query_progress/')
def query_progress():
    return jsonify({'state': 1, 'msg': round(bar.get(), 2)})


@app.route(url + 'initialize/')
def initialize():
    """未避免因强制终止程序使得sqlite中的进度不为1而导致无法进行融合，请求本服务以将其置为0"""
    bar.create()
    return jsonify(({'state': 1, 'msg': '初始化状态成功'}))


if __name__ == '__main__':
    # 生成环境下启动使用`gunicorn -c gunicorn_config.py app:app`
    app.run('0.0.0.0')
