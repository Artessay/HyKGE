from sentence_transformers import SentenceTransformer
import itertools
import numpy as np
# from FlagEmbedding import FlagReranker

# 加载模型，使用 GPU 加速
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-de")
reranker_model = SentenceTransformer("BAAI/bge-reranker-v2-m3")        # refer: https://huggingface.co/BAAI/bge-reranker-v2-m3

def get_embeddings_batch(texts):
    emb = embedding_model.encode(texts)
    return emb.tolist()

def reranker_chains(origin_data, chains, top_n=10):
    # 原始数据存储在 doc_list 中
    doc_list = [origin_data]
    
    # 知识列表直接使用传入的 chains
    knowledge_list = chains

    # 对 doc_list 和 knowledge_list 进行笛卡尔积操作
    compared_list = [list(pair) for pair in itertools.product(doc_list, knowledge_list)]
    
    # 计算每对的分数
    scores = reranker_model.compute_score(compared_list)
    scores = np.array(scores)
    
    # 如果每个知识项的得分都是独立的，可以跳过 reshape
    if len(scores.shape) > 1:
        scores = np.max(scores, axis=1)
    
    print(len(scores))
    
    # 按照分数排序，取前 TOP_N 个
    question_sim_idx = np.argsort(scores)[-min(top_n, len(chains)):]
    
    # 根据排序结果更新新的 gold_paths
    new_gold_paths = [chains[item] for item in question_sim_idx.tolist()]
    
    return new_gold_paths