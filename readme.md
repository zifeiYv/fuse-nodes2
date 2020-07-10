# 1. python基础环境部署
1.1 推荐安装`Anaconda`，版本为`4.8.2`，下载地址为：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh

1.2 如果可以联网，直接利用`wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh`
下载；如果不可联网，则手动下载后拷贝至目标服务器

1.3 执行`sh Anaconda3-2020.02-Linux-x86_64.sh`进行安装，安装位置保持默认即可，询问设置时选择[yes]

1.4 安装完成后可能需要重新导入一下配置文件，执行`source ~/.bashrc`

1.5 执行`conda -V`，如果打印`conda 4.8.2`，则说明安装成功

# 2. python依赖安装
2.1 所有的依赖包已经离线下载到`./pkgs`文件夹内，只需要执行`pip install --no-index --find-links=./pkgs -r requirements.txt`即可完成安装

# 3. 参数配置
3.1 需要配置的参数在`./config_files`文件夹内，一共有两个，分别是：`application.cfg`和`neo4j-structure.cfg`。在对应的文件中，已经解释了每个参数的含义以及设置方式，按照说明执行即可。

# 4. 执行融合命令
4.1 融合的入口为`run_cmd.py`脚本，以命令行参数的形式传入生成的融合图的标签，基本用法如下：
```python
 python run_cmd.py --new-label xxxx
```
    
`xxxx`为标签名称。

另外，可以执行`python run_cmd.py -h`查看帮助。

如果待融合图较大，那么可以指定`--multi`参数为`1`，启动多进程来加速运算（测试中，可能出现意料之外的错误，默认为关闭）。
