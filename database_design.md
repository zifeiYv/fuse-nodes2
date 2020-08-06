# 0、术语
- **实体实例**：一个具体的客观对象，例如，一个人、一张桌子、一个变压器等。
- **实体类**：是实体实例的抽象表示，例如，一个人是人类（实体类）的实例，那么称"人类"为实体类。
- **系统**：一般指的是实体实例所在的数据库，例如，人力资源系统、财务管理系统。
- **融合的属性**：指的是融合依赖的属性，例如，依据人的姓名、身高、收入等判断两个记录在不同系统中的实体数据是否为同一个人。
- **迁移的属性**：指的是融合后新生成的节点从原节点获取的属性，例如，可以设定把原系统中"人"的姓名迁移到融合后的节点上。

# 1、说明
算法所需要的参数存储在关系型数据库中（MySQL），在开始执行算法前，程序首先从给定的数据库中获取对应参数并进行核验，参数合法后才进行后续计算。

# 2、数据库设计
## 2.1 表结构
存储参数的表一共有两张：`fuse_config_table1`和`fuse_config_2`，各表的结构如下：
```sql
-- ----------------------------
-- Table structure for fuse_config_table1
-- ----------------------------
DROP TABLE IF EXISTS `fuse_config_table1`;
CREATE TABLE `fuse_config_table1` (
  `id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '主键',
  `task_id` varchar(64) DEFAULT NULL COMMENT '融合任务的id',
  `system_label` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '待融合的系统的标签',
  `entity_label` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '待融合的系统中的实体的标签',
  `pros_for_fuse` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '依赖哪些属性来融合',
  `pros_for_transfer` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '融合后将哪些属性迁移到新生成的节点上',
  `entity_level` int DEFAULT NULL COMMENT '实体的级别',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

SET FOREIGN_KEY_CHECKS = 1;
```
```sql
-- ----------------------------
-- Table structure for fuse_config_table2
-- ----------------------------
DROP TABLE IF EXISTS `fuse_config_table2`;
CREATE TABLE `fuse_config_table2` (
  `id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '唯一标识',
  `from_id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '引用fuse_config_table1中的主键，关系的开始实体',
  `to_id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '引用fuse_config_table1中的主键，关系的终止实体',
  `task_id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '融合任务的id',
  `rel_label` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '关系的标签',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

SET FOREIGN_KEY_CHECKS = 1;
```

存储融合结果的表为`fuse_result_table`，其结构如下：
```sql
-- ----------------------------
-- Table structure for fuse_result_table
-- ----------------------------
DROP TABLE IF EXISTS `fuse_result_table`;
CREATE TABLE `fuse_result_table` (
  `id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '主键',
  `task_id` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '任务的id',
  `subgraph_id` int DEFAULT NULL COMMENT '子图的id，在每个任务中是从1开始自增长的',
  `entity_type` varchar(8) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '实体的类别标签',
  `value` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '记录原始实体的id，由英文逗号连接',
  `sorted_sys` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL COMMENT '记录系统选择的顺序',
  `run_time` datetime DEFAULT NULL COMMENT '执行时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

SET FOREIGN_KEY_CHECKS = 1;
```

## 2.2 表说明
### 2.2.1 `fuse_config_table1`

- 每一次调用融合代码，称为执行了一个任务，每个任务都具有一个唯一的任务标识（`task_id`);
- 程序会根据传入的`task_id`从`fuse_config_table1`表中获取该任务需要的所有非关系类的配置信息；
- 要成功执行一个融合任务，需要满足一些必须的约束条件：

    - 系统数量（`system_label`）必须多于1个
    - 每个系统下待融合的实体类（`entity_label`）的数量必须多于1个
    - 用于融合的属性（`pros_for_fuse`）必须多于1个，如果有多个，则它们之间用英文逗号连接
    - 对于同等级的实体，其用于融合的属性（`pros_for_fuse`）的数量必须相等
    - 对于融合后的实体，必须至少有一个迁移属性（`pros_for_fuse`），如果有多个，则它们之间用英文逗号连接
    

### 2.2.2 `fuse_config_table2`

对于一个待融合的系统而言，如果存在多个需要融合的实体类，那么必须定义实体类之间的关系。注意，实体类之间的关系总是由低级指向高级的（1级实体指向2级实体）。

### 2.2.3 `fuse_result_table`

执行完成一个融合任务之后，在将融合结果生成至图数据库之前，需要将结果保存在关系型数据库中。

# 3、配置文件
为了获取参数，需要指定参数表所在的数据库。限定数据库的类型为MySQL，其连接信息在`./config_files/applicaiton.cfg`中定义，section为`mysql`，option为`mysql_cfg`。