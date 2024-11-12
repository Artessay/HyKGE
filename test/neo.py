import time
import json
from py2neo import Node,Relationship,Graph,Path,Subgraph
from py2neo import NodeMatcher,RelationshipMatcher
# 连接neo4j数据库，输入地址、用户名、密码
# graph= Graph("bolt://localhost:7687",auth=("neo4j", "limengyao20191123"))
graph = Graph("bolt://localhost:7688", auth=("neo4j", "jiangxinke"))


def find_all_paths_between_nodes(node1_name, node2_name):
    # query = (
    #     f"MATCH (n1 {{name: '{node1_name}'}}), (n2 {{name: '{node2_name}'}}), "
    #     "p = allShortestPaths((n1)-[*..]-(n2)) "
    #     "RETURN p"
    #    )
    start_time = time.time()  # 记录开始时间    
    entity_list = ['低钠血症', '急性肾衰竭', '高钾血症', '血清氯化物', '少尿'] # ['低钠血症', '急性肾衰竭', '高钾血症', '血清氯化物', '少尿', '急性全自主神经失调症', '体内水钠潴留', '维生素A', '血便', '复方电解质', '尿频', '肾损伤', '小儿低氯性氮质血症综合征', '失盐', '心悸', '肾脏受累', '无尿']
    # paths = [
    #     graph.run(
    #         f"MATCH (n1 {{name: '{node1_name}'}}), (n2 {{name: '{node2_name}'}}), "
    #             "p = (n1)-[*1..2]-(n2) "
    #             "RETURN p"
    #     ).data()
    #     for node1_name in entity_list
    #     for node2_name in entity_list
    # ]

    # 在数据可以全部装进内存的情况下，仅仅两跳查询时，Neo4j的速度并没有原来的路径查询快
    
    # for node1_name in entity_list:
    #     for node2_name in entity_list:
    #         query = (
    #             f"MATCH (n1 {{name: '{node1_name}'}}), (n2 {{name: '{node2_name}'}}), "
    #             "p = (n1)-[*1..2]-(n2) "
    #             "RETURN p"
    #         )
    #         paths = graph.run(query).data()
    query = """
    CALL apoc.periodic.iterate(
      'UNWIND ["肺部感染", "青霉素"] AS name1 UNWIND ["肺部感染", "青霉素"] AS name2 RETURN DISTINCT name1, name2 WHERE name1 < name2',
      'MATCH (n1 {name: $name1}), (n2 {name: $name2})
       MATCH p = (n1)-[*1..2]-(n2)
       RETURN p',
      {batchSize:100, parallel:true}
    )
    """
    paths = graph.run(query).data()
    
    end_time = time.time()  # 记录结束时间
    print(f"查询时间：{end_time - start_time}秒")  # 输出查询所需时间
    
    return paths


#    query = (
#        'MATCH (from:node), (to:node) where from.name="肺部感染" and to.name="青霉素" '
#         'CALL apoc.algo.allSimplePaths(from, to, "rel>", 7) YIELD path RETURN  path'
#    )
#     # query = (
#     #     f"MATCH path = (n1 {{name: '{node1_name}'}})-[*]->(n2 {{name: '{node2_name}'}}) "
#     #     "RETURN path"
#     # )
#    query = (
#         f"MATCH (n1 {{name: '{node1_name}'}}), (n2 {{name: '{node2_name}'}}), "
#         "p = (n1)-[*..]-(n2) "
#         "RETURN p"
#     )
   


print(graph)
node1_name = "肺部感染"  # 第一个节点的名称
node2_name = "青霉素"  # 第二个节点的名称

paths = find_all_paths_between_nodes(node1_name, node2_name)


# paths = graph.run('MATCH (from:node), (to:node) where from.name="肺部感染" and to.name="青霉素" '
#              'CALL apoc.algo.allSimplePaths(from, to, "rel>", 7) YIELD path RETURN  path').data()

for path in paths:
    print(path['p'])

def find_all_paths_between_entity_list(entity_list):
    all_paths = []
    for start in entity_list:
        print("start: ",start)
        for end in entity_list:
            if start != end:
                print("end: ",end)
                paths_dict = {"start":start,"end":end}
                paths = find_all_paths_between_nodes(start, end)
                paths_dict["paths"] = []
                for path in paths:
                    print(path['p'])
                    paths_dict["paths"].append(path['p'])
                all_paths.append(paths_dict)
    return all_paths

# with open('data/MMCU_test_2819_second.json', 'r') as file:
#     data_list = json.load(file)
# result_list = []
# for index in range(0,len(data_list)):
#     print("=========================================")
#     print(index)
#     data ={}
#     data["id"] = data_list[index]["id"]
#     data["question"] = data_list[index]["question"]
#     data["answer"] = data_list[index]["answer"]
#     data["question_type"] = data_list[index]["type"]
#     data["res"] = data_list[index]["res"]
#     data["entity_list"] = data_list[index]["entity"]
#     entity_list = data_list[index]["entity"]
#     if len(entity_list)  < 2:
#             data["all_path"] = []
#     else:
#         data["all_path"] = find_all_paths_between_entity_list(entity_list)
#     # print(data["all_path"])
#     result_list.append(data)
#     with open('data/MMCU_triplets.json', 'w') as file:
#         json.dump(result_list, file, ensure_ascii=False, indent=4)