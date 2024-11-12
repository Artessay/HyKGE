#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2024/3/25 17:12
# @File: document_search.py

import timeit
import logging
from typing import List, Optional

if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..document_retrieval import DOCLinker

class DOCRetriever():
    def __init__(
            self,
            logger: logging.Logger = None,
            encode_method: Optional[str] = 'GTE'
    ) -> None:
        self.logger = logger

        # sentence linking module
        logger.info("[Sentence Linker] loading ...") if logger else None
        self.entity_linker = DOCLinker.DocumentLinker(encode_method=encode_method)
        logger.info("[Sentence Linker] load success") if logger else None

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
            f"[Sentence Linker] Elapsed Time (s): {timeit.default_timer() - start_entity_linker: .4f}") if logger else None
        logger.debug('Sentence List: ', entity_list) if logger else None

        return entity_list
