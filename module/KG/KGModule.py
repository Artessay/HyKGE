from config import Config
from models.medical_ner.medical_ner import *
from sentence_transformers import SentenceTransformer
from models.entity_linking import KGRetriever
from utils.search_web import EncyclopediaDictionary
from utils.cut_stop_words import *
from utils.singleton import Singleton
from utils.cut_stop_words import *
from utils.prompt_build import *
import torch
import datetime


@Singleton
class KGModule:
    def __init__(self, config: Config = None):
        self.ENCODING_CHOICE = config.configKG.ENCODING_CHOICE
        self.USE_PATH = config.configKG.USE_PATH
        self.max_hop = config.configKG.max_hop
        self.knowledge_format_choice = config.configKG.knowledge_format_choice
        self.WEB_ENABLE = config.configKG.WEB_ENABLE
        self.WEB_Filter_SIM_THRESHOLD = config.configKG.WEB_Filter_SIM_THRESHOLD

        self.web_search = EncyclopediaDictionary()
        self.stop_words_cache = load_stop_words()
        # ENCODING_Choice_list = ["CME", "Piccolo", "GTE"]
        if self.ENCODING_CHOICE == "Piccolo":
            self.encoding_model = SentenceTransformer('./models/embedding_model/piccolo_checkpoint/')
        if self.ENCODING_CHOICE == "GTE":
            self.encoding_model = SentenceTransformer('./models/embedding_model/gte_checkpoint/')
        self.kg_retriever = KGRetriever(logger=None, encode_method=self.ENCODING_CHOICE)

        self.config = config

    def processKnowledge(self, ner_res, sim_threshold):
        # 2. Fetch Embedding
        gold_triplets, fact_triplets = [], []

        if ner_res != []:
            entity_path_dict, kg_entity_list = self.kg_retriever.inference(ner_res)

            path_list = []
            path_fake_list = []
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
                                    path_fake_list.append(path)

            # 如果是只有一个实体，直接返回
            if self.USE_PATH:
                if len(entity_path_dict) == 1:
                    self.knowledge_format_choice = 'triplets'
                    gold_triplets = fact_triplets
                else:
                    self.knowledge_format_choice = 'paths'
                    gold_triplets = path_list

            print("gold triplets length:", len(gold_triplets))
            print(len(path_list), len(path_fake_list))
            print(gold_triplets)

        return gold_triplets, kg_entity_list

    def processDescription(self, entity_list, origin_data):
        if self.WEB_ENABLE:
            entity_description = self.web_search.search_entity_descriptions(entity_list)
        else:
            entity_description = {}
        # print("entity description", entity_description)

        # 过滤description：
        new_entity_description = {}
        if entity_description == {}:
            pass
        else:
            filtered_title = chinese_word_cut(origin_data, self.stop_words_cache)
            question_title_embedding = torch.tensor(
                self.encoding_model.encode(filtered_title, normalize_embeddings=True)).cuda().unsqueeze(0)
            filtered_web_list = []
            for entity_name in entity_description.keys():
                filtered_web = chinese_word_cut(entity_description[entity_name], self.stop_words_cache)
                filtered_web_list.append(filtered_web)

            web_embedding = torch.tensor(
                self.encoding_model.encode(filtered_web_list, normalize_embeddings=True)).cuda()
            question_sim_matrix = question_title_embedding @ web_embedding.T

            keys = list(entity_description.keys())
            for idx, question_sim in enumerate(question_sim_matrix.squeeze(0)):
                if question_sim > self.WEB_Filter_SIM_THRESHOLD:
                    new_entity_description[keys[idx]] = entity_description[keys[idx]]

            entity_description = new_entity_description
        return entity_description

    def processDescription_ragent(self, entity_list, origin_data):
        if self.WEB_ENABLE:
            entity_description = self.web_search.search_entity_descriptions(entity_list)
        else:
            entity_description = {}

        return entity_description

    def fetchKGPath(self, mention_list, user_query, sim_threshold):
        # KG Module for Knowledge
        print("KG Module start")
        starttime = datetime.datetime.now()
        gold_triplets, entity_list = self.processKnowledge(mention_list, sim_threshold)
        endtime = datetime.datetime.now()
        print("KG Module finished in :", (endtime - starttime).seconds, " seconds")

        # KG Module for Description
        print("KG Module(description) start")
        starttime = datetime.datetime.now()
        entity_description = self.processDescription(mention_list, user_query)
        endtime = datetime.datetime.now()
        print("KG Module(description) finished in :", (endtime - starttime).seconds, " seconds")

        RAG_prompt = build_RAGent_prompt(gold_triplets, entity_description, user_query, self.config)

        return RAG_prompt