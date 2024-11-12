import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.configServer = ServerConfig(config=config["server"])
        self.configHO = HOConfig(config=config["HO_module"])
        self.configNER= NERConfig(config=config["NER_module"])
        self.configKG = KGConfig(config=config["KG_module"])
        self.configDOC = DOCConfig(config=config["DOC_module"])
        self.configFilter = FilterConfig(config=config["Filter_module"])
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())


class ServerConfig:
    def __init__(self,config):
        self.POOL_NUM = config["POOL_NUM"]
        self.SERVER_HOST = config["SERVER_HOST"]
        self.SERVER_PORT = config["PORT"]
        self.history_epoch = config["history_epoch"]
        self.RETURN_PROMPT = config["RETURN_PROMPT"]
        self.DOCUMENT_SOURCE = config["DOCUMENT_SOURCE"]

class HOConfig:
    def __init__(self,config):
        self.HyDE = config["HyDE"]
        self.query_expansion = config["query_expansion"]
        self.LLM_type = config["LLM_type"]
        self.hyde_max_token_num = config["hyde_max_token_num"]

class NERConfig:
    def __init__(self,config):
        self.NER_Choice = config["NER_Choice"]
        self.device = config["device"]

class KGConfig:
    def __init__(self,config):
        self.KG_ENABLE = config["KG_ENABLE"]
        self.USE_PATH = config["USE_PATH"]
        self.knowledge_format_choice = config["knowledge_format_choice"]
        self.max_hop = config["max_hop"]
        self.ENCODING_CHOICE = config["ENCODING_CHOICE"]
        self.WEB_ENABLE = config["description"]["WEB_ENABLE"]
        self.WEB_Filter_SIM_THRESHOLD = config["description"]["WEB_Filter_SIM_THRESHOLD"]

class DOCConfig:
    def __init__(self,config):
        self.DOC_TOP_K = config["TopK"]
        self.ENCODING_CHOICE = config["DOC_ENCODING_CHOICE"]
        self.DOC_SIM_THRESHOLD = config["DOC_SIM_THRESHOLD"]


class FilterConfig:
    def __init__(self,config):
        self.FILTER_ENABLE = config["FILTER_ENABLE"]
        self.TOP_SIM = config["TOP_SIM"]
        self.use_reranker = config["use_reranker"]
        self.USE_CHUNK = config["USE_CHUNK"]


        """
{
  "server":{
    "POOL_NUM":1,
    "SERVER_HOST":"0.0.0.0",
    "PORT":8195,
    "history_epoch":5,
    "RETURN_PROMPT":true,
    "DOCUMENT_SOURCE":"KG"    # KG or DOC
  },
  "HO_module":{
    "HyDE":true,
    "query_expansion":false,
    "LLM_type":"GPT3.5",
    "hyde_max_token_num":500
  },
  "NER_module":{
    "NER_Choice":"W2NER",
    "device":0
  },
  "KG_module":{
    "KG_ENABLE":true,
    "USE_PATH":true,
    "knowledge_format_choice":"paths",
    "max_hop":3,
    "ENCODING_CHOICE":"GTE",
    "description":{
      "WEB_ENABLE":true,
      "WEB_Filter_SIM_THRESHOLD":0.2
    }
  },
  "Filter_module":{
    "FILTER_ENABLE":true,
    "TOP_SIM":11,
    "use_reranker":true,
    "USE_CHUNK":true
  }
}

        """

