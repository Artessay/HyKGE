from config import Config
from utils.cut_stop_words import *
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import itertools
import torch
from utils.singleton import Singleton

@Singleton
class FilterModule:
    def __init__(self,config: Config = None):
        self.FILTER_ENABLE = config.configFilter.FILTER_ENABLE
        self.use_reranker = config.configFilter.use_reranker
        self.knowledge_format_choice = config.configKG.knowledge_format_choice
        self.TOP_SIM = config.configFilter.TOP_SIM
        self.HyDE = config.configHO.HyDE
        self.USE_CHUNK = config.configFilter.USE_CHUNK
        self.ENCODING_CHOICE = config.configKG.ENCODING_CHOICE

        self.stop_words_cache = load_stop_words()
        # ENCODING_Choice_list = ["CME", "Piccolo", "GTE"]
        if self.ENCODING_CHOICE == "Piccolo":
            self.encoding_model = SentenceTransformer('./models/embedding_model/piccolo_checkpoint/')
        if self.ENCODING_CHOICE == "GTE":
            self.encoding_model = SentenceTransformer('./models/embedding_model/gte_checkpoint/')
        if self.use_reranker:
            self.reranker = FlagReranker(r'./models/reranker_model/bge_reranker_large/',
                            use_fp16=True)

    
    def process(self,gold_triplets,origin_data,hyde_data):
        if gold_triplets == [] or not self.FILTER_ENABLE:
            pass
        else:
            if not self.use_reranker:
                # 题干筛选相似度
                relation_embedding_list = []
                for relation_in_gold_triplet in gold_triplets:
                    if self.knowledge_format_choice == 'triplets':
                        relation_embedding = torch.tensor(self.encoding_model.encode(
                            relation_in_gold_triplet['S'] + relation_in_gold_triplet['P'] + relation_in_gold_triplet['O'],
                            normalize_embeddings=True)).cuda().unsqueeze(0)
                    elif self.knowledge_format_choice == 'paths':
                        item = ','.join(relation_in_gold_triplet)
                        relation_embedding = torch.tensor(self.encoding_model.encode(item)).cuda().unsqueeze(0)

                    relation_embedding_list.append(relation_embedding)
                relation_embedding_list = torch.stack(relation_embedding_list).squeeze(1)

                # 题干的embedding
                filtered_title = chinese_word_cut(origin_data, self.stop_words_cache)
                if self.HyDE:
                    filtered_hyde = chinese_word_cut(hyde_data, self.stop_words_cache)
                question_title_embedding = torch.tensor(
                    self.encoding_model.encode(filtered_title, normalize_embeddings=True)).cuda().unsqueeze(0)
                question_sim = relation_embedding_list @ question_title_embedding.T
                question_sim_idx = torch.topk(question_sim, k=min(self.TOP_SIM, len(gold_triplets)), dim=0)

                new_gold_triplet = []
                for item in question_sim_idx.indices.tolist():
                    new_gold_triplet.append(gold_triplets[item[0]])
                gold_triplets = new_gold_triplet

            else:  # reranker
                compared_list = []
                filtered_title = chinese_word_cut(origin_data, self.stop_words_cache)
                doc_list = [filtered_title]
                knowledge_list = []
                if self.HyDE and self.USE_CHUNK:
                    filtered_hyde = chinese_word_cut(hyde_data, self.stop_words_cache)
                    doc_list += chunk_text(filtered_hyde,35,15)
                    doc_list += [filtered_hyde]

                for relation_in_gold_triplet in gold_triplets:
                    if self.knowledge_format_choice == 'triplets':
                        knowledge_list.append(relation_in_gold_triplet['S'] + relation_in_gold_triplet['P'] +
                                            relation_in_gold_triplet['O'])
                    elif self.knowledge_format_choice == 'paths':
                        # item = ','.join(relation_in_gold_triplet)
                        item = ','.join(str(item) for item in relation_in_gold_triplet)
                        knowledge_list.append(item)
                compared_list = [list(pair) for pair in itertools.product(doc_list, knowledge_list)]
                scores = self.reranker.compute_score(compared_list)
                scores = np.array(scores).reshape(len(knowledge_list), len(doc_list))
                scores = np.max(scores, axis=1)
                print(len(scores))
                question_sim_idx = np.argsort(scores)[-min(self.TOP_SIM, len(gold_triplets)):]
                new_gold_triplet = []
                for item in question_sim_idx.tolist():
                    new_gold_triplet.append(gold_triplets[item])
                gold_triplets = new_gold_triplet
        return gold_triplets
