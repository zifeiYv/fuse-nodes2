# -*- coding: utf-8 -*-
"""
#-------------------------------------------------------------------#
#                    Project Name : 实体融合                         #
#                                                                   #
#                       File Name : self_check.py                   #
#                                                                   #
#                          Author : Jiawei Sun                      #
#                                                                   #
#                          Email : j.w.sun1992@gmail.com            #
#                                                                   #
#                      Start Date : 2020/07/16                      #
#                                                                   #
#                     Last Update : 2020/07/16                      #
#                                                                   #
#-------------------------------------------------------------------#
"""
from self_check import check
from utils import Nodes
from tqdm import tqdm
FUSE_AND_CREATE = True  # 融合一个子图，创建一个子图


if __name__ == '__main__':
    print("必要的检查...")
    check()
    print("一切正常\n")

    print("开始融合")
    from fuse import fuse_root_nodes, fuse_other_nodes, \
        BASE_SYS_ORDER, LABEL, create_node_and_rel, delete_old

    print("删除旧的融合结果...")
    delete_old('merge')
    print("删除完成\n")

    root_res_df = fuse_root_nodes()
    print(root_res_df)
    if root_res_df is None:
        print("根节点融合后无结果，无法继续执行")
    else:
        print("根节点融合完成，开始融合子图")
        base_ent_lab = LABEL[BASE_SYS_ORDER[0]].iloc[0]
        for i in tqdm(range(len(root_res_df))):
            if i != 0:
                continue

            print(root_res_df.iloc[i].to_list())
            node = Nodes(base_ent_lab, root_res_df.iloc[i].to_list())
            fuse_other_nodes(1, node, BASE_SYS_ORDER)  # 执行之后，node包含了创建一个子图所需要的完整信息
            if FUSE_AND_CREATE:
                create_node_and_rel(node)
        print("创建新图完成")
