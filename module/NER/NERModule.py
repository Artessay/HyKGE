from config import Config
from utils.prompt_build import *
from utils.cut_stop_words import *
from models.W2NER.baseline.W2NER_predictor import load_W2NER_model, predict
from models.medical_ner.medical_ner import *
from utils.singleton import Singleton

@Singleton
class NERModule:
    def __init__(self,config: Config = None):
        self.NER_Choice = config.configNER.NER_Choice
        self.stop_words_cache = load_stop_words()
        # NER_Choice_list = ["CME", "LLM", "W2NER", "Jieba"]
        # NER_Choice = NER_Choice_list[2]
        if self.NER_Choice == "W2NER":
            self.W2NER, self.W2NER_config, self.W2NER_args = load_W2NER_model(device='cuda:0')
        elif self.NER_Choice == "CME":
            self.my_pred = medical_ner(use_cuda=True, device='cuda:0')


    
    def process(self,data):
        # 1. NER(data) => type:entity_name
        if self.NER_Choice == 'LLM':
            res = LLM_process_NER(data)
            res = res.split(',')
            sim_threshold = 0.9
        elif self.NER_Choice == 'CME':
            filtered_text = chinese_word_cut(data, self.stop_words_cache)
            print("分词结果：", filtered_text)
            res = my_pred.predict_sentence(filtered_text)
            if res is None:
                res = []
            else:
                res = list(set(res))
            sim_threshold = 0.9
        elif self.NER_Choice == 'W2NER':
            filtered_text = chinese_word_cut(data, self.stop_words_cache)
            print(type(filtered_text))
            print(len(filtered_text))
            res = predict(filtered_text[0:511], self.W2NER, self.W2NER_config, self.W2NER_args)

            print(res)
            sim_threshold = 0.6
        else:
            filtered_text = chinese_word_cut(data, self.stop_words_cache)
            print("分词结果：", filtered_text)
            res = filtered_text.split(',')[:-1]

        res = list(set(res))
        print("抽取结果:", res)
        return res,sim_threshold