# -*- coding: utf-8 -*-
"""
File Name  : app
Author     : Jiawei Sun
Email      : j.w.sun1992@gmail.com
Start Date : 2020/07/16
Describe   :
    应用的启动入口
"""
from self_check import check, get_paras
from flask import Flask, jsonify, request, json
from werkzeug.exceptions import HTTPException
from concurrent.futures import ThreadPoolExecutor
from progressbar import ProgressBar
from fuse import main_fuse
import traceback
import sys

url = '/entity_fuse/'
executor = ThreadPoolExecutor(1)

app = Flask(__name__)


@app.route(url, methods=['POST'])
def func():
    print("必要的检查...")
    task_id = request.json['task_id']
    assert task_id is not None, '必须传入任务id'
    try:
        merged_label = check(task_id)
    except Exception:
        print(__file__, sys._getframe().f_lineno, 'ERROR:', traceback.print_exc())
        return jsonify({'state': 0, 'msg': "配置文件非法，查看控制台输出"})
    print("一切正常")
    bar = ProgressBar(merged_label)
    bar.create()
    if bar.get() not in (1.0, 0.0):
        return jsonify({'state': 0, "msg": "当前有正在执行的任务，请等待其完成后重试"})

    # main_fuse(task_id)
    executor.submit(main_fuse, task_id)
    return jsonify({"state": 1, "msg": "正在后台进行融合任务"})


@app.route(url + 'query_progress/', methods=['POST'])
def query_progress():
    task_id = request.json['task_id']
    _, _, _, merged_label = get_paras(task_id)
    bar = ProgressBar(merged_label)
    return jsonify({'state': 1, 'msg': round(bar.get(), 2)})


@app.route(url + 'initialize/', methods=['POST'])
def initialize():
    """未避免因强制终止程序使得sqlite中的进度不为1而导致无法进行融合，请求本服务以将其置为0"""
    task_id = request.json['task_id']
    _, _, _, merged_label = get_paras(task_id)
    bar = ProgressBar(merged_label)
    bar.create()
    return jsonify(({'state': 1, 'msg': '初始化状态成功'}))


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


if __name__ == '__main__':
    # 生成环境下启动使用`gunicorn -c gunicorn_config.py app:app`
    app.run('0.0.0.0')
