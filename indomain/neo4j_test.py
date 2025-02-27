import pandas as pd
from py2neo import Graph, Node, Relationship

# 设置CSV文件的根目录
path_root = r"/home/qrh/data/code/Clinical-and-Diagnostic-Reasoning/KG_construct/data/"
files = ['KG_append_0.csv', 'KG_append_1.csv', 'KG_append_2.csv', 'KG_append_3.csv', 'KG_append_4.csv']

# 读取文件并合并
data_frames = [pd.read_csv(path_root + file) for file in files]
triples = pd.concat(data_frames, ignore_index=True)

# 设置图数据库连接
graph = Graph("bolt://localhost:7687", auth=("neo4j", "limengyao20191123"))
graph.delete_all()


# 遍历DataFrame行并创建图节点和关系
for index, row in triples.iterrows():
    print(index)

    start_node = Node("my_entity", name=row["head"])
    end_node = Node("my_entity", name=row["tail"])
    relation = Relationship(start_node, row["relation"], end_node)

    # 合并节点到图中，确保唯一性
    graph.merge(start_node, "my_entity", "name")
    graph.merge(end_node, "my_entity", "name")

    # 创建关系
    graph.create(relation)

print("Nodes and relationships have been created in the graph.")