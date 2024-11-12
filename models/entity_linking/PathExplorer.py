import os
import numpy as np
import pandas as pd

class PathExplorer():
    def __init__(self) -> None:
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # self.relation_columns = ['relation_id', 'entity1', 'predicate', 'entity2']
        relation_dataset: pd.DataFrame = pd.read_pickle(f'{dir_path}/resource/relation.pkl')

        # construct adjacency dict
        self.adj_list: dict = PathExplorer._construct_adjacency_list(relation_dataset)
    
    def explore_paths(self, entity_list: list):
        # start level paths
        return {start: self._find_paths_for_source(start, entity_list) for start in entity_list}

    def _find_paths_for_source(self, src: str, entity_list: list):
        # init paths
        entity_path = {entity : ([], []) for entity in entity_list}

        entity_set = set(entity_list)
        entity_set.remove(src)
        
        for (src_entity, src_triple) in self.adj_list[src]:
            if src_entity in entity_set:
                # reduce duplication
                path = [src, src_entity]
                if path not in entity_path[src_entity][0]:
                    entity_path[src_entity][0].append(path)
                    entity_path[src_entity][1].append([src_triple])
            else:
                for dst in entity_set:
                    for (dst_entity, dst_triple) in self.adj_list[dst]:
                        if src_entity == dst_entity:
                            # reduce duplication
                            path = [src, src_entity, dst]
                            if path not in entity_path[dst][0]:
                                entity_path[dst][0].append(path)
                                entity_path[dst][1].append([src_triple, dst_triple])

        # special case for path[src][src]
        entity_path[src] = (
            [[src, entity] for (entity, _) in self.adj_list[src]],
            [[triple] for (_, triple) in self.adj_list[src]]
        )
        
        return entity_path

    @staticmethod
    def _construct_adjacency_list(relation_dataset):
        src: np.ndarray = relation_dataset['entity1'].to_numpy()
        dst: np.ndarray = relation_dataset['entity2'].to_numpy()
        triples: np.ndarray = np.array(PathExplorer._construct_triples(relation_dataset))
        
        num_nodes = len(src)
        assert num_nodes == len(dst) and num_nodes == len(triples)
        
        # initiate adjacency list
        adj_list = {}

        # the adjacency vector stores edges for each node (source or destination), undirected
        # adj_list, dict of list, where each element is a list of triple tuple (enity, relation)
        adj_info_src = np.vstack((dst, triples)).T  # adjacency information for source nodes
        adj_info_dst = np.vstack((src, triples)).T  # adjacency information for destination nodes

        for src_node, adj_info in zip(src, adj_info_src):
            if src_node in adj_list:
                adj_list[src_node].append(tuple(adj_info))
            else:
                adj_list[src_node] = [tuple(adj_info)]

        for dst_node, adj_info in zip(dst, adj_info_dst):
            if dst_node in adj_list:
                adj_list[dst_node].append(tuple(adj_info))
            else:
                adj_list[dst_node] = [tuple(adj_info)]

        return adj_list

    @staticmethod
    def _construct_triples(relation_dataset):
        edge_triples = [
            {
                'S': getattr(row, 'entity1'),
                'P': getattr(row, 'predicate'),
                'O': getattr(row, 'entity2')
            }
            for row in relation_dataset.itertuples(index=False)
        ]

        return edge_triples
    
if __name__ == '__main__':
    import os
    import sys
    import timeit
    from pprint import pprint
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path_explorer = PathExplorer()
    print('initial finished')

    # Test 01

    entity_list = ['肺部感染', '青霉素', '败血症']
    result = path_explorer._find_paths_for_source('肺部感染', entity_list)
    pprint(result['青霉素'])
    print()

    paths = path_explorer.explore_paths(entity_list)
    pprint(paths['败血症']['肺部感染'])
    print()

    # Test 02

    entity_list = ['低钠血症', '急性肾衰竭', '高钾血症', '血清氯化物', '少尿', '急性全自主神经失调症', '体内水钠潴留', '维生素A', '血便', '复方电解质', '尿频', '肾损伤', '小儿低氯性氮质血症综合征', '失盐', '心悸', '肾脏受累', '无尿']
    
    start_time = timeit.default_timer()
    paths = path_explorer.explore_paths(entity_list)
    explore_time = timeit.default_timer() - start_time
    
    assert len(paths['低钠血症']['低钠血症'][0]) == len(path_explorer.adj_list['低钠血症'])
    
    # node paths
    pprint(paths['高钾血症']['维生素A'][0])
    print()
    pprint(paths['高钾血症']['维生素A'][1])
    print()

    import itertools
    path_count = 0
    for entity_pair in itertools.permutations(entity_list, r=2):
        start, end = entity_pair
        try:
            (path, _) = paths[start][end]
            path_count += len(path)
        except Exception as e:
            print("  error:", str(e))

    # print(paths['维生素A']['维生素A'])
    print(f"Entity count: {len(entity_list)}, Path count: {path_count}, Time(s): {explore_time}")

    # Test 03
    entity_list = ["肺部感染", "青霉素"]
    
    start_time = timeit.default_timer()
    paths = path_explorer.explore_paths(entity_list)
    explore_time = timeit.default_timer() - start_time

    print(f"Entity count: {len(entity_list)}, Time(s): {explore_time}")
