import requests
import json
from .prompt_template import *
from config import Config


def build_RAG_prompt(gold_triplets, entity_description, origin_data, config: Config = None, ):
    if gold_triplets == []:
        if entity_description == {}:
            RAG_prompt = origin_data
        else:
            if config.configHO.HyDE:
                # RAG_prompt = hyde_discription_prompt_build(entity_description, hyde_data, origion_data)
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
            else:
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
    else:  # 非首次问诊，用 gold prompt
        # 1. 所有三元组
        # RAG_prompt = prompt_build(fact_triplets, gold_triplets, entity_description, origion_data)
        # 2. Overlap
        if config.configHO.HyDE:
            RAG_prompt = gold_prompt_build(gold_triplets, entity_description, origin_data,
                                           config.configKG.knowledge_format_choice)
            # RAG_prompt = hyde_gold_prompt_build(gold_triplets, entity_description, hyde_data, origion_data)
        else:
            RAG_prompt = gold_prompt_build(gold_triplets, entity_description, origin_data,
                                           config.configKG.knowledge_format_choice)

        # 3. LLM组织单个三元组
        # RAG_prompt = LLM_COT_gold_prompt_build(gold_triplets, entity_description, origion_data)
        # 4. LLM组织多个三元
        # RAG_prompt = LLM_process_gold_prompt_build(gold_triplets, entity_description, origion_data)
    print("返回Prompt", RAG_prompt)
    return RAG_prompt


def build_RAGent_prompt(gold_triplets, entity_description, origin_data, config: Config = None, ):
    if gold_triplets == []:
        if entity_description == {}:
            RAG_prompt = origin_data
        else:
            if config.configHO.HyDE:
                # RAG_prompt = hyde_discription_prompt_build(entity_description, hyde_data, origion_data)
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
            else:
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
    else:  # 非首次问诊，用 gold prompt
        # 1. 所有三元组
        # RAG_prompt = prompt_build(fact_triplets, gold_triplets, entity_description, origion_data)
        # 2. Overlap
        if config.configHO.HyDE:
            RAG_prompt = agent_gold_prompt_build(gold_triplets, entity_description, origin_data,
                                           config.configKG.knowledge_format_choice)
            # RAG_prompt = hyde_gold_prompt_build(gold_triplets, entity_description, hyde_data, origion_data)
        else:
            RAG_prompt = agent_gold_prompt_build(gold_triplets, entity_description, origin_data,
                                           config.configKG.knowledge_format_choice)

        # 3. LLM组织单个三元组
        # RAG_prompt = LLM_COT_gold_prompt_build(gold_triplets, entity_description, origion_data)
        # 4. LLM组织多个三元
        # RAG_prompt = LLM_process_gold_prompt_build(gold_triplets, entity_description, origion_data)
    print("返回Prompt", RAG_prompt)
    return RAG_prompt


def build_ICL_prompt(gold_triplets, entity_description, sentence_list, origin_data, config: Config = None, ):
    if gold_triplets == []:
        if entity_description == {}:
            RAG_prompt = origin_data
        else:
            if config.configHO.HyDE:
                # RAG_prompt = hyde_discription_prompt_build(entity_description, hyde_data, origion_data)
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
            else:
                RAG_prompt = discription_prompt_build(entity_description, origin_data)
    else:  # 非首次问诊，用 gold prompt
        # 1. 所有三元组
        # RAG_prompt = prompt_build(fact_triplets, gold_triplets, entity_description, origion_data)
        # 2. Overlap
        if config.configHO.HyDE:
            RAG_prompt = ICL_gold_prompt_build(gold_triplets, entity_description, sentence_list, origin_data,
                                               config.configKG.knowledge_format_choice)
            # RAG_prompt = hyde_gold_prompt_build(gold_triplets, entity_description, hyde_data, origion_data)
        else:
            RAG_prompt = ICL_gold_prompt_build(gold_triplets, entity_description, sentence_list, origin_data,
                                               config.configKG.knowledge_format_choice)

        # 3. LLM组织单个三元组
        # RAG_prompt = LLM_COT_gold_prompt_build(gold_triplets, entity_description, origion_data)
        # 4. LLM组织多个三元
        # RAG_prompt = LLM_process_gold_prompt_build(gold_triplets, entity_description, origion_data)
    print("返回Prompt", RAG_prompt)
    return RAG_prompt


def build_DOC_prompt(sentence_list, origin_data, config: Config = None, ):
    if config.configHO.HyDE:
        RAG_prompt = DOC_gold_prompt_build(sentence_list, origin_data)
        # RAG_prompt = hyde_gold_prompt_build(gold_triplets, entity_description, hyde_data, origion_data)
    else:
        RAG_prompt = DOC_gold_prompt_build(sentence_list, origin_data)
    print("返回Prompt", RAG_prompt)
    return RAG_prompt


def build_HyDE_prompt(data, query_expansion):
    """
    [In Use] HyDE or QE prompt used in HO Module

    Args:
        data (_type_): _description_
        query_expansion (_type_): _description_

    Returns:
        _type_: _description_
    """
    if query_expansion:
        return QE_IN_template + data + "\"\n}"
    else:
        return HyDE_IN_template.format(data)


def build_COK_prompt(data, stage=1):
    """
    [! In Use] Baseline COK

    Args:
        data (_type_): _description_
        stage (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if stage == 1:
        return COK_REASONING_GENERATION_template.format(data)
    else:
        return COK_RATIONALE_CORRECTION_template.format(data[0], data[1])


def prompt_build(fact_triplets, gold_triplets, entity_description, query):
    """
    [! In Use] 最后的prompt

    Args:
        fact_triplets (_type_): _description_
        gold_triplets (_type_): _description_
        entity_description (_type_): _description_
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    RAG_prompt = "请用以下检索到的医疗事实三元组知识辅助你回答该任务。最需要关注的知识有："
    gold_triplet_pre = ""
    for fact in gold_triplets:
        gold_triplet_pre += "{}和{}的关系是{},".format(fact['S'], fact['O'], fact['P'])

    RAG_prompt = RAG_prompt + gold_triplet_pre + '; 还有一些知识也能辅助你判断：'

    triplet_pre = ""
    for fact in fact_triplets:
        triplet_pre += ",{}和{}的关系是{}".format(fact['S'], fact['O'], fact['P'])

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        RAG_prompt = RAG_prompt + triplet_pre

        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            RAG_prompt = RAG_prompt + str(entity_description_pre)

    RAG_prompt = RAG_prompt + '请结合上述知识和自己的认知，分点详细地回答问题：' + query

    return RAG_prompt


def gold_prompt_build(gold_triplets, entity_description, query, knowledge_format_choice):
    """
    [In Use] 最终返回的prompt,包括knowlededge and description

    Args:
        gold_triplets (_type_): _description_
        entity_description (_type_): _description_
        query (_type_): _description_
        knowledge_format_choice (_type_): _description_

    Returns:
        _type_: _description_
    """
    gold_triplet_pre = ""
    web_prompt = ""

    if knowledge_format_choice == 'triplets':
        for fact in gold_triplets:
            gold_triplet_pre += "{}和{}的关系是{},".format(fact['S'], fact['O'], fact['P'])
    elif knowledge_format_choice == 'paths':
        gold_triplet_pre += '以下知识为解释路径：'
        for fact in gold_triplets:
            modified_string = ""
            for item in fact:
                tlist = str(item).split(":")
                if tlist[0] == "i":
                    modified_string += "->"
                    modified_string += tlist[1]
                    modified_string += "->"
                elif tlist[0] == "o":
                    modified_string += "<-"
                    modified_string += tlist[1]
                    modified_string += "<-"
                else:
                    modified_string += tlist[0]

            gold_triplet_pre += "{},".format(modified_string)

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            web_prompt = web_prompt + str(entity_description_pre)

    RAG_prompt = prompt_template.format(gold_triplet_pre + '\n' + web_prompt, query)
    return RAG_prompt


def agent_gold_prompt_build(gold_triplets, entity_description, query, knowledge_format_choice):
    """
    [In Use] 最终返回的prompt,包括knowlededge and description

    Args:
        gold_triplets (_type_): _description_
        entity_description (_type_): _description_
        query (_type_): _description_
        knowledge_format_choice (_type_): _description_

    Returns:
        _type_: _description_
    """
    gold_triplet_pre = ""
    web_prompt = ""

    if knowledge_format_choice == 'triplets':
        for fact in gold_triplets:
            gold_triplet_pre += "{}和{}的关系是{},".format(fact['S'], fact['O'], fact['P'])
    elif knowledge_format_choice == 'paths':
        gold_triplet_pre += '以下知识为解释路径：'
        for fact in gold_triplets:
            modified_string = ""
            for item in fact:
                tlist = str(item).split(":")
                if tlist[0] == "i":
                    modified_string += "->"
                    modified_string += tlist[1]
                    modified_string += "->"
                elif tlist[0] == "o":
                    modified_string += "<-"
                    modified_string += tlist[1]
                    modified_string += "<-"
                else:
                    modified_string += tlist[0]

            gold_triplet_pre += "{},".format(modified_string)

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            web_prompt = web_prompt + str(entity_description_pre)

    # RAG_prompt = prompt_template.format(gold_triplet_pre + '\n' + web_prompt, query)
    return gold_triplet_pre


def ICL_gold_prompt_build(gold_triplets, entity_description, sentence_ICL, query, knowledge_format_choice):
    """
    [In Use] 最终返回的prompt,包括knowlededge and description

    Args:
        gold_triplets (_type_): _description_
        entity_description (_type_): _description_
        query (_type_): _description_
        knowledge_format_choice (_type_): _description_

    Returns:
        _type_: _description_
    """
    gold_triplet_pre = ""
    web_prompt = ""

    if knowledge_format_choice == 'triplets':
        for fact in gold_triplets:
            gold_triplet_pre += "{}和{}的关系是{},".format(fact['S'], fact['O'], fact['P'])
    elif knowledge_format_choice == 'paths':
        gold_triplet_pre += '以下知识为解释路径：'
        for fact in gold_triplets:
            modified_string = ""
            for item in fact:
                tlist = str(item).split(":")
                if tlist[0] == "i":
                    modified_string += "->"
                    modified_string += tlist[1]
                    modified_string += "->"
                elif tlist[0] == "o":
                    modified_string += "<-"
                    modified_string += tlist[1]
                    modified_string += "<-"
                else:
                    modified_string += tlist[0]

            gold_triplet_pre += "{},".format(modified_string)

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            web_prompt = web_prompt + str(entity_description_pre)

    # RAG_prompt = ICL_prompt_template_tjy.format(background_1=(gold_triplet_pre + '\n' + web_prompt), background_2=sentence_ICL,ques=query)
    RAG_prompt = ICL_prompt_template.format((gold_triplet_pre + '\n' + web_prompt), sentence_ICL, query)

    return RAG_prompt


def DOC_gold_prompt_build(sentence_ICL, query):
    # RAG_prompt = ICL_prompt_template_tjy.format(background_1=(gold_triplet_pre + '\n' + web_prompt), background_2=sentence_ICL,ques=query)
    RAG_prompt = ICL_prompt_template.format((""), sentence_ICL, query)

    return RAG_prompt


def hyde_gold_prompt_build(gold_triplets, entity_description, hyde_out, query):
    """
    [! In Use] 最终返回的prompt，包括hyde_out，指导大模型订正输出

    Args:
        gold_triplets (_type_): _description_
        entity_description (_type_): _description_
        hyde_out (_type_): _description_
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    gold_triplet_pre = ""
    web_prompt = ""
    for fact in gold_triplets:
        gold_triplet_pre += "{}和{}的关系是{},".format(fact['S'], fact['O'], fact['P'])

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            web_prompt = web_prompt + str(entity_description_pre)

    RAG_prompt = HyDE_OUT_template.format(gold_triplet_pre + '\n' + web_prompt, hyde_out, query)
    return RAG_prompt


def discription_prompt_build(entity_description, query):
    """
    [In Use] 最终返回的promopt，只补充description知识

    Args:
        entity_description (_type_): _description_
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    web_prompt = ""

    for key, value in entity_description.items():
        entity_description_pre = "{}的定义是{};".format(key, value)
        web_prompt = web_prompt + str(entity_description_pre)
    RAG_prompt = prompt_template.format(web_prompt, query)

    return RAG_prompt


def hyde_discription_prompt_build(entity_description, hyde_out, query):
    """
    [! In Use] 最终返回的只有descripiton的prompt，包括hyde_out，指导大模型订正输出

    Args:
        entity_description (_type_): _description_
        hyde_out (_type_): _description_
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    web_prompt = ""
    for key, value in entity_description.items():
        entity_description_pre = "{}的定义是{};".format(key, value)
        web_prompt = web_prompt + str(entity_description_pre)
    RAG_prompt = HyDE_OUT_template.format(web_prompt, hyde_out, query)

    return RAG_prompt


def talk_to_LLM(context="大家好，我是三年二班的周杰伦", temperature=0.5, response_num=3, max_tokens=100):
    url = 'http://localhost:8080/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "string",
        "messages": [
            {
                "role": "user",
                "content": context
            }
        ],
        "do_sample": True,
        "temperature": temperature,
        "n": response_num,
        "max_tokens": max_tokens,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_data = response.json()
    message = []
    if 'detail' in response_data:
        print("Encounted Validation Error---->msg:{}  type:{}".format(response_data['detail'][0]['msg'],
                                                                      response_data['detail'][0]['type']))
    else:
        for msg in response_data['choices']:
            message.append(msg['message']['content'])
    return message


# 处理响应数据
# response_data 包含了服务器返回的JSON数据，你可以根据需要进行处理


### BELOW ARE TEST####

def LLM_COT_gold_prompt_build(gold_triplets, entity_description, query):
    from multiprocessing.connection import Client

    client = Client(('localhost', 8100))

    RAG_prompt = "请用以下检索到的医疗事实知识辅助你回答该任务。最需要关注的知识有："
    gold_triplet_pre = ""

    for fact in gold_triplets:
        COT_Prompt = ("我现在给你一段三元组，请帮我组织成一句话，请不要输出任何与该三元组无关多余的话，无法组织就返回空值。示例如下：\
                      （北京大学，坐落于，北京市），组织成：”北京大学坐落于北京市“。\
                      请仿照以上形式，帮我组织以下三元组，只输出这句话，别的一个字也不要说：({},{},{})".format(fact['S'],
                                                                                                           fact['P'],
                                                                                                           fact['O']))
        client.send(COT_Prompt)
        LLM_COT_Prompt = client.recv()  # 等待接受数据
        # print(data)
        # time.sleep(1)

        gold_triplet_pre += LLM_COT_Prompt

    RAG_prompt = RAG_prompt + gold_triplet_pre

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        RAG_prompt = RAG_prompt + '。在所检索的知识中，'

        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            RAG_prompt = RAG_prompt + str(entity_description_pre)

    RAG_prompt = RAG_prompt + '请结合上述知识和自己的认知，分点详细地回答问题：' + query

    return RAG_prompt


def LLM_process_gold_prompt_build(gold_triplets, entity_description, query):
    from multiprocessing.connection import Client

    client = Client(('localhost', 8100))

    RAG_prompt = "请用以下检索到的医疗事实知识辅助你回答该任务。最需要关注的知识有："
    gold_triplet_pre = ""

    COT_Prompt = "我现在给你一堆三元组，请帮整理和总结成一段话，请不要输出任何与该三元组无关多余的话，无法组织就返回空值。示例如下：\
                  [（北京大学，坐落于，北京市），（北京大学，原名，燕京大学）]，组织成：”北京大学原名燕京大学，坐落于北京市“。\
                  请仿照以上形式，帮我组织以下三元组，只输出这些知识，别的一个字也不要说：" + str(gold_triplets)
    client.send(COT_Prompt)
    LLM_COT_Prompt = client.recv()  # 等待接受数据

    gold_triplet_pre += LLM_COT_Prompt

    RAG_prompt = RAG_prompt + gold_triplet_pre

    if entity_description == {}:
        pass
    else:
        # 生成关于三元组的提示
        RAG_prompt = RAG_prompt + '。在所检索的知识中，'

        for key, value in entity_description.items():
            entity_description_pre = "{}的定义是{};".format(key, value)
            RAG_prompt = RAG_prompt + str(entity_description_pre)

    RAG_prompt = RAG_prompt + '请结合上述知识和自己的认知，分点详细地回答问题：' + query

    return RAG_prompt


# RAG_prompt = prompt_build(fact_triplets, gold_triplets, entity_description)
# print(RAG_prompt)


def LLM_process_NER(query):
    query = str(query).replace("\n", "")
    Instruction = "\"指令\":从给定的文本中抽取出可能的跟医学相关的实体.输出按照json格式{\"output\":[entity,entity]}请严格按照格式输出,只抽取药物、疾病、症状、治疗方案等实体。"
    Example = '''
"示例":
"输入":我感觉肚子疼和脑袋痛,感觉感冒了，昨天吃了罗红霉素
"输出"：{"output":["肚子疼","脑袋痛","感冒","罗红霉素"]}
"示例":
"输入":利福平是一种有机化合物，是一种所属利福霉素家族的一种广谱抗生素药物，主要用于治疗结核病、脑膜炎和金黄色葡萄球菌感染，外用可治疗沙眼等。
"输出"：{"output":["利福平","利福霉素","结核病","脑膜炎","沙眼"]}
下面是这次的输入，请给出输出，其他一个字也不要多说。
'''

    COT_Prompt = Instruction + Example + "\"输入\"：" + str(query) + "\n\"输出\"："
    # print(COT_Prompt)
    response = talk_to_LLM(context=COT_Prompt, temperature=0.6)  # 默认返回三次
    ner = []
    # print(response)
    for re in response:
        re = str(re).replace("\n", "")
        try:
            re = json.loads(re)
        except json.JSONDecodeError:
            continue
        if 'output' in re:
            for entity in re['output']:
                if entity not in ner:
                    ner.append(entity)  # 取三次回答的并集，不对json格式做修正，不满足回答格式的就抛弃

    if len(ner) == 0:
        COT_Prompt = Instruction + "请注意输出格式。" + Example + "\"输入\"：" + str(query) + "\"输出\"："
        # print(COT_Prompt)
        response = talk_to_LLM(context=COT_Prompt, temperature=0, response_num=1)  # 如果ner为空，为了避免格式错误被抛弃的答案，重新问一次
        try:
            response = json.loads(response[0])
        except json.JSONDecodeError:
            ner = []
        if 'output' in response:
            for entity in response['output']:
                if entity not in ner:
                    ner.append(entity)

    res = ''
    for e in ner:
        res = res + e + ","
    print("LLM抽取结果:")
    print(res)
    return res


def build_KO_prompt(query, gold_triplets, entity_description, sentence_list):
    """
    [In Use] KO Module

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return KO_IN_template.format(query, entity_description, sentence_list, gold_triplets)


def build_KO_RAG_prompt(KO_knowledge, origin_data, config: Config = None, ):
    RAG_prompt = KO_knowledge_prompt_template.format(KO_knowledge, origin_data)

    print("返回Prompt", RAG_prompt)
    return RAG_prompt

# from cut_stop_words import *
# from W2NER.baseline.W2NER_predictor import load_W2NER_model, predict
# W2NER, W2NER_config = load_W2NER_model(device='cuda:0')
# #talk_to_LLM()
# list = []
# list.append('以下是中国药师考试中初级药士考试的一道单项选择题，请分析每个选项，并最后给出答案。\n服用维生素B6可避免或减轻周围神经炎发生的是\nA. 异烟肼\nB. 利福平\nC. 卡那霉素\nD. 乙胺丁醇\nE. 吡嗪酰胺')
# list.append("我今天有些胸闷，早晨便血，刚刚吃了阿莫西林")
# for data in list:
#     print(data)
#     res=LLM_process_NER(data)
#     stop_words_cache = load_stop_words()
#     filtered_text = chinese_word_cut(data, stop_words_cache)
#     res = predict(filtered_text, W2NER, W2NER_config)
#     print("W2NER抽取结果：")
#     print(res)
#     print()

