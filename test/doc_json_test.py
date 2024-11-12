#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2024/4/25 15:15
# @File: doc_json_test.py

import pickle
import pandas as pd
import json
doc_path = r'./models/document_retrieval/resource/others/document_gte_tjy.pkl'

entity_dataset = pd.read_pickle(doc_path)

processed_list = [{"text": text[:2048]} for text in entity_dataset]

json_data = json.dumps(processed_list, ensure_ascii=False, indent=4)

# 将JSON字符串存储到txt文件中
with open("rag_document.json", "w", encoding="utf-8") as file:
    file.write(json_data)