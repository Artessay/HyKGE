from config import Config
from sentence_transformers import SentenceTransformer
from models.document_retrieval import DOCRetriever
from utils.cut_stop_words import *
from utils.singleton import Singleton
from utils.cut_stop_words import *
import torch


@Singleton
class DOCModule:
    def __init__(self, config: Config = None):
        self.DOC_TOP_K = config.configDOC.DOC_TOP_K
        self.ENCODING_CHOICE = config.configDOC.ENCODING_CHOICE
        self.DOC_SIM_THRESHOLD = config.configDOC.DOC_SIM_THRESHOLD

        # ENCODING_Choice_list = ["CME", "Piccolo", "GTE"]
        if self.ENCODING_CHOICE == "Piccolo":
            self.encoding_model = SentenceTransformer('./models/embedding_model/piccolo_checkpoint/')
        if self.ENCODING_CHOICE == "GTE":
            self.encoding_model = SentenceTransformer('./models/embedding_model/gte_checkpoint/')
        self.doc_retriever = DOCRetriever(logger=None, encode_method=self.ENCODING_CHOICE)

    def processKnowledge(self, data, sim_threshold=0.6):
        sentence_list = self.doc_retriever.inference(data, sim_threshold)

        return sentence_list