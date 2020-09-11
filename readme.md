# 1. python基础环境部署
 1. 推荐安装`Anaconda`，版本为`4.8.2`，下载地址为：
 ```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh
```

2.  如果可以联网，直接利用`wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh`
下载；如果不可联网，则手动下载后拷贝至目标服务器。

3. 执行`sh Anaconda3-2020.02-Linux-x86_64.sh`进行安装，安装位置保持默认即可，询问设置时选择[yes]。

4. 安装完成后**可能**需要重新导入一下配置文件，执行`source ~/.bashrc`。

5. 执行`conda -V`，如果打印`conda 4.8.2`，则说明安装成功。

# 2. python依赖安装
1. 在线安装只需要执行`pip install -r requirements.txt`即可。

2. 离线安装步骤要复杂一些：
    
    2.1 先要配置与目标服务器操作系统相同的虚拟环境，并在其中安装`python 3.7.x`
    
    2.2 利用`pip`制作依赖包的whl文件，命令为：`pip wheel --wheel-dir=/path/to/wheels -r requirements.txt`
    
    2.3 将存储所有whl文件的文件夹和`requirements.txt`一起上传至目标服务器上，执行安装命令：
    ```
    pip install --no-index --find-links=/path/to/your/wheels -r requirements.txt
    ```

# 3. 参数配置
1. 需要配置的参数在`./config_files`文件夹内，只有一个文件。在文件中，已经解释了每个参数的含义以及设置方式，按照说明执行即可。

# 4. 启动服务
1. 通过`gunicorn`启动http服务的命令为：`gunicorn -c gunicorn_config.py app:app`。

2. `gunicorn_config.py`中定义了服务相关的配置，其中监听的端口默认为5000。

# 5. 接口说明
1.  `http://127.0.0.1:5000/entity_fuse/`

    请求方式：GET  
    接口说明：执行融合任务

2. `http://127.0.0.1:5000/entity_fuse/query_progress/`
    
    请求方式：GET  
    接口说明：查询当前任务的执行进度

3. `http://127.0.0.1:5000/entity_fuse/initialize/` 

    请求方式：GET  
    接口说明：当服务被终止时仍有在执行任务，请求此接口将该任务的进度重置

