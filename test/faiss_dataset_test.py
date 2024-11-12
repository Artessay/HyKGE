#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2024/3/28 09:24
# @File: faiss_dataset_test.py

import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


doc_path = r'./models/document_retrieval/resource/others/document_gte_tjy.pkl'
faiss_path = r'./models/document_retrieval/resource/others/faiss_index_gte_tjy.bin'
embedding_path = r'./models/document_retrieval/resource/checkpoint_gte/'

entity_dataset = pd.read_pickle(doc_path)

faiss_index = faiss.read_index(faiss_path)

model = SentenceTransformer(embedding_path)


def search(query_sentence, model, index, entity_dataset, k=5):
    # 生成查询句子的嵌入向量
    query_embedding = model.encode([query_sentence], normalize_embeddings=True)

    # 使用FAISS索引进行查询
    distances, indices = index.search(query_embedding, k)

    # 获取最相似的句子及其相似度
    similar_sentences = []
    for distance, idx in zip(distances[0], indices[0]):
        similar_sentences.append((entity_dataset[idx], 1 - distance))

    return similar_sentences


# 示例查询
query = "B疱疹病毒感染的病因是什么？:(一)发病原因BV颗粒直径约"  # 替换为你的查询句子
# query = "下列有关鳃弓的描述，错误的是：A: 由间充质增生形成 B: 人胚第4周出现, C: 相邻鳃弓之间为鳃沟, D: 共5对鳃弓 E: 位于头部两侧"  # 替换为你的查询句子
# query = "按照三阶梯用药原则，适用于中度癌痛的是可待因、丁丙诺啡、布洛芬 氢吗啡酮、右丙氧芬、安度芬 二氢可待因、消炎痛、美沙酮 可待因、氨酚待因、强痛定 芬太尼透皮贴剂、曲马多、奇曼丁"  # 替换为你的查询句子
similar_sentences = search(query, model, faiss_index, entity_dataset)

# 打印查询结果
for sentence, similarity in similar_sentences:
    print(f"Sentence: {sentence}, Similarity: {similarity:.2f}")

