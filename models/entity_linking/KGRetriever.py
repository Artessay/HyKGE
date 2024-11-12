import timeit
import logging
from typing import List, Optional

if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..entity_linking import EntityLinker
from ..entity_linking import PathExplorer


class KGRetriever():
    def __init__(
            self,
            logger: logging.Logger = None,
            encode_method: Optional[str] = 'GTE'
    ) -> None:
        self.logger = logger

        # entity linking module
        logger.info("[Entity Linker] loading ...") if logger else None
        self.entity_linker = EntityLinker(encode_method=encode_method)
        logger.info("[Entity Linker] load success") if logger else None

        # path retriever module
        logger.info("[Path Explorer] loading ...") if logger else None
        self.path_explorer = PathExplorer()
        logger.info("[Path Explorer] load success") if logger else None

    def inference(
            self,
            mention_list: List[str],
            entity_linking_threshold: Optional[float] = 0.6
    ):
        logger = self.logger

        # 1. link mentions to KG entities
        start_entity_linker = timeit.default_timer()
        entity_list = self.entity_linker.entity_linking(mention_list, entity_linking_threshold)
        logger.info(
            f"[Entity Linker] Elapsed Time (s): {timeit.default_timer() - start_entity_linker: .4f}") if logger else None
        logger.debug('Entity List: ', entity_list) if logger else None

        # 2. retrieve paths in KG
        start_path_explorer = timeit.default_timer()
        paths = self.path_explorer.explore_paths(entity_list)
        logger.info(
            f"[Path Explorer] Elapsed Time (s): {timeit.default_timer() - start_path_explorer: .4f}") if logger else None

        # TODO

        return paths, entity_list


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    mention_list = ['尿', '无尿', '心脏', '少尿', '高钾血症', '紧急电解质失调', '血症', '电解质', '氯离子', '低氯血症',
                    '血', '肾脏功能受损', '体内离子', '电解质失调', '肾脏', '急性肾衰竭', '低钠血症', '维生素',
                    '体内钠离子异常排泄']
    kg_retriever = KGRetriever(logger=logger)

    entity_path_dict, kg_entity_list = kg_retriever.inference(mention_list)

    path_list = []
    for left_node in kg_entity_list:
        for right_node in kg_entity_list:
            if entity_path_dict[left_node][right_node] == []:
                pass
            else:
                (paths, descriptions) = entity_path_dict[left_node][right_node]
                for inx, path in enumerate(paths):
                    description = descriptions[inx]
                    if len(path) == 1:
                        path_list.append(path)
                    else:
                        # print(entity_path_dict[left_node][right_node][0])
                        hash_S = dict()
                        hash_O = dict()
                        path = []
                        for item in description:
                            if len(path) == 0:
                                path.append(item['S'])
                            if path[-1] == item['S']:
                                path.append("i:" + item['P'])
                                path.append(item['O'])
                            elif path[-1] == item['O']:
                                path.append("o:" + item['P'])
                                path.append(item['S'])

                            if item['S'] in hash_S.keys():
                                hash_S[item['S']] += 1
                            else:
                                hash_S[item['S']] = 1
                            if item['O'] in hash_O.keys():
                                hash_O[item['O']] += 1
                            else:
                                hash_O[item['O']] = 1

                        in_S = 0
                        out_S = 0
                        for jtem in hash_S.values():
                            if jtem != 1:
                                in_S += jtem
                        for jtem in hash_O.values():
                            if jtem != 1:
                                out_S += jtem

                        if in_S + out_S == 2 or in_S + out_S == 0:
                            # print(in_S, out_S)
                            path_list.append(path)
                        else:
                            pass

    print(len(path_list))
    assert len(path_list) == 3757