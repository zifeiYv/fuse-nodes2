# -*- coding: utf-8 -*-
"""应用的启动入口

.. note::
    在测试时，可以执行``python app.py``启动服务；在正式环境中，必须执行``gunicorn -c gunicorn_config.py app:app``启动服务。

.. note::
    服务在任何情况下均返回json对象，其中，键"state"对应的取值及其解释如下：
    0：正常结束
    1：程序报错
    2：程序正在运行
"""
from self_check import check
from flask import Flask, jsonify, request, json
from werkzeug.exceptions import HTTPException
from concurrent.futures import ThreadPoolExecutor
# from progressbar import ProgressBar
from fuse import main_fuse
from log_utils import gen_logger
import traceback
import sys
import os

if not os.path.exists('./logs'):
    os.makedirs('./logs')
url = '/entity_fuse/'
executor = ThreadPoolExecutor(1)

app = Flask(__name__)


@app.route(url, methods=['POST'])
def func():
    task_id = request.json['task_id']
    logger = gen_logger(task_id)
    logger.info("必要的检查...")
    try:
        _ = check(task_id)
    except Exception as e:
        logger.error(e)
        return jsonify({'state': 1, 'msg': "配置文件非法，查看日志输出"})
    logger.info("一切正常")

    main_fuse(task_id)
    # executor.submit(main_fuse, task_id)
    return jsonify({"state": 0, "msg": "正在后台进行融合任务"})


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
