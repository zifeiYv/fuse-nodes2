# -*- coding: utf-8 -*-
"""
========================================
Configuration file for *gunicorn*.
========================================
You are not recommended to start this app using *Flask server* in a production deployment
for the sake of efficiency and security. Meanwhile, *gunicorn* is a good choice.

This python script contains the configuration settings of *gunicorn*. Python syntax must
be followed when you modify it.

Once set, you can start the server using::

    gunicorn -c gunicorn_config.py app:app

or::

    gunicorn -c gunicorn_config.py -D app:app

if you want to run it background.

For more information, see `Gunicorn settings docs <https://docs.gunicorn.org/en/latest/settings.html#config>`_.
"""
from os import cpu_count

# The socket to bind.
bind = '0.0.0.0:5000'

# The number of worker processes for handling requests.
# A positive integer generally in the ``2-4*cpu_nums`` range.
workers = cpu_count()

# The number of worker threads for handling requests.
# Run each worker with the specified number of threads.
# A positive integer generally in the ``2-4*cpu_nums`` range.
threads = 2

# The Access log file to write to.
accesslog = './logs/info.log'
# The Error log file to write to.
errorlog = './logs/error.log'
# The granularity of Error log outputs.
loglevel = 'warning'
