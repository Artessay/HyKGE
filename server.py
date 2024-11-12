import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3,6'

import argparse
import multiprocessing
from utils.prompt_build import *
import datetime
from module import ModuleManager

# todo: mention==1 => triplets;  path => triplets

this_var_is_shit = 0

# knowledge_format_index = 1
# knowledge_format = ['triplets', 'paths']
# knowledge_format_choice = knowledge_format[knowledge_format_index]

# Load args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=r'./config/DOC_example.json')

args = parser.parse_args()

from config.config import Config

config = Config(args)
ModuleManagerInstance = ModuleManager()
ModuleManagerInstance.setup(config)
print(ModuleManagerInstance)
print(ModuleManagerInstance.mapper())

SERVER_HOST = config.configServer.SERVER_HOST
SERVER_PORT = config.configServer.SERVER_PORT


def mapping_func(data_dict, histroy_query_buffer, config: Config):
    # global histroy_query_buffer        # todo 影响多进程
    # 单例模式
    print(ModuleManagerInstance)
    print(ModuleManagerInstance.mapper())
    HO_module, NER_module, KG_module, DOC_module, Filter_module, KO_module = ModuleManagerInstance.mapper()

    qa_epoch = data_dict['clear']
    origin_data = data_dict['query']

    # NOTE STAGE: BEFORE RETRIEVAL
    # HO Module
    print("HO Module start")
    starttime = datetime.datetime.now()
    data, histroy_query_buffer = HO_module.process(origin_data, qa_epoch, histroy_query_buffer)
    endtime = datetime.datetime.now()
    print("HO Module finished in :", (endtime - starttime).seconds, " seconds")
    starttime = endtime

    # TODO: ADD DOCUMENT RAG HERE

    if 'KG' in config.configServer.DOCUMENT_SOURCE:
        # NER Module
        print("NER Module start")
        ner_res, sim_threshold = NER_module.process(data)
        endtime = datetime.datetime.now()
        print("NER Module finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

        # NOTE STAGE: RETRIEVAL
        # KG Module for Knowledge
        print("KG Module start")
        gold_triplets, entity_list = KG_module.processKnowledge(ner_res, sim_threshold)
        endtime = datetime.datetime.now()
        print("KG Module finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

        # KG Module for Description
        print("KG Module(description) start")
        # entity_description = KG_module.processDescription(entity_list,origin_data)
        entity_description = KG_module.processDescription(ner_res, origin_data)
        endtime = datetime.datetime.now()
        print("KG Module(description) finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

    if 'DOC' in config.configServer.DOCUMENT_SOURCE:
        # TODO ADD DOC => 注意和后面的文件名称对应关系。
        print("DOC Module start")
        sentence_list = DOC_module.processKnowledge(data)

        if 'KG' in config.configServer.DOCUMENT_SOURCE:
            gold_triplets.extend(sentence_list)
            # sentence_list
            pass
        else:
            gold_triplets = sentence_list
            entity_description = None
        endtime = datetime.datetime.now()
        print("DOC Module finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime
        print(sentence_list)
        pass
    else:
        pass

    # NOTE STAGE: AFTER RETRIEVAL
    # Filter Module
    print("Filter Module start")
    gold_triplets = Filter_module.process(gold_triplets, origin_data, data)

    endtime = datetime.datetime.now()
    print("Filter Module finished in :", (endtime - starttime).seconds, " seconds")
    starttime = endtime

    # Knowledge Organize Module
    # print("KO Module start")
    # starttime = datetime.datetime.now()
    # KO_knowledge = KO_module.process(gold_triplets, entity_description, sentence_list, origin_data)
    # endtime = datetime.datetime.now()
    # print("KO Module finished in :", (endtime - starttime).seconds," seconds")
    # starttime = endtime

    # note support multi-round q&a 0509
    if 'now_query' in data_dict.keys():
        origin_data = data_dict['now_query']        # query是历史的对话，query是当前的对话


    # fixme 恢复注释
    # Without KG
    if not config.configKG.KG_ENABLE:
        gold_triplets = []

    # Without return prompt
    if not config.configServer.RETURN_PROMPT:
        return [gold_triplets, entity_description, origin_data], histroy_query_buffer

    # 4. Build prompts
    if 'DOC' in config.configServer.DOCUMENT_SOURCE and 'KG' in config.configServer.DOCUMENT_SOURCE:
        print(gold_triplets)
        print(entity_description)
        RAG_prompt = build_ICL_prompt(gold_triplets, entity_description, sentence_list, origin_data, config)

    elif 'KG' in config.configServer.DOCUMENT_SOURCE:
        RAG_prompt = build_RAG_prompt(gold_triplets, entity_description, origin_data, config)
    else:
        RAG_prompt = build_DOC_prompt(sentence_list, origin_data, config)
    # RAG_prompt = RAG_prompt + '\n最后请以：”答案是：xx选项“的方式输出结果。'

    # todo 这一块记得整理一下
    # RAG_prompt = build_KO_RAG_prompt(KO_knowledge,origin_data,config)

    return RAG_prompt, histroy_query_buffer


def do_socket(conn, addr, config):
    histroy_query_buffer = []
    data_dict = None
    try:
        data_dict = conn.recv()  # Fetching data from the client
        print(data_dict)
    except EOFError:
        pass
    except Exception as e:
        print('Socket Error', e)

    if data_dict is None:
        out = ""
    else:
        try:
            out, histroy_query_buffer = mapping_func(data_dict, histroy_query_buffer, config)  # Return output
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("Execution Error:", e)
            out = data_dict['query']  # str(e)
            histroy_query_buffer = []

    try:
        conn.send(out)
    except Exception as e:
        print('Socket Error:', e)

    try:
        conn.close()
        print('Connection close.', addr)
    except:
        print('close except')


def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')


def run_server(host, port, config: Config):
    from multiprocessing.connection import Listener
    multiprocessing.set_start_method('spawn')  # 设置启动方法为'spawn' => CUDA必须使用spawn格式
    server_sock = Listener((host, port))

    print("Server running...", host, port)

    pool = multiprocessing.Pool(config.configServer.POOL_NUM)  # POOL NUM

    while True:
        # 接受一个新连接:
        conn = server_sock.accept()
        addr = server_sock.last_accepted
        print('Accept new connection', addr)

        # 创建进程来处理TCP连接:
        pool.apply_async(func=do_socket, args=(conn, addr, config,), error_callback=err_call_back)


if __name__ == '__main__':
    run_server(SERVER_HOST, SERVER_PORT, config)

#     {
#   "server":{
#     "POOL_NUM":1,
#     "SERVER_HOST":"0.0.0.0",
#     "PORT":8195,
#     "history_epoch":5,
#     "RETURN_PROMPT":true
#   },
#   "HO_module":{
#     "HyDE":true,
#     "query_expansion":false,
#     "LLM_type":"GPT3.5",
#     "hyde_max_token_num":500
#   },
#   "NER_module":{
#     "NER_Choice":"W2NER",
#     "device":0
#   },
#   "KG_module":{
#     "KG_ENABLE":true,
#     "USE_PATH":true,
#     "knowledge_format_choice":"paths",
#     "max_hop":3,
#     "ENCODING_CHOICE":"GTE",
#     "description":{
#       "WEB_ENABLE":true,
#       "WEB_Filter_SIM_THRESHOLD":0.2
#     }
#   },
#   "Filter_module":{
#     "FILTER_ENABLE":true,
#     "TOP_SIM":11,
#     "use_reranker":true,
#     "USE_CHUNK":true
#   }
# }