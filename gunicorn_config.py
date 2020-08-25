# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : gunicorn_config.py              #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/27                      #
#                                                                   #
#                     Last Update : 2020/07/27                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
# 启动方式
# gunicorn -c gunicorn_config.py app:app
#
# 监听的ip与端口
bind = '0.0.0.0:5000'

# 进程数量
# workers = 4

# 日志处理
accesslog = './logs/info.log'

errorlog = './logs/error.log'


loglevel = 'warning'
