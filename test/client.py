#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2023/10/23 13:46
# @File: scipt2.py

import time

from multiprocessing.connection import Client
client = Client(('0.0.0.0', 8205))

query_prompt_1 = "以下是中国{exam_type}中{exam_class}考试的一道{exam_subject}相关的{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}"
query_prompt_2 = "以下是中国{exam_type}中{exam_class}考试的一道{exam_subject}{question_type}，请直接输出正确选项，多的一句话也不要回答。\n{question}\n{option_str}"
def get_query(da):
    da["exam_type"] = da["exam_type"]
    da["exam_class"] = da["exam_class"]
    da["exam_subject"] = da["exam_subject"]
    da["question_type"] = da["question_type"]
    da["question"] = da["question"]
    da["option_str"] = "\n".join(
        [f"{k}. {v}" for k, v in da["option"].items() if len(v) > 0 and v!=" "]
    )
    query = query_prompt_1.format_map(da)
    print("Answer is:", da["answer"])
    return query
# 。请直接输出正确选项，多的一句话也不要回答。
title_idx = 5
# 题1 D: RAG错; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei错；GPT4-RAG对；无Hyde对；空行x；
# 题2 B: RAG错; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei对；GPT4-RAG对；无Hyde错；空行；
# 题3 E: RAG对; RAW错; GPT4错; Reranker-RAG对；openai_xiaobei错；GPT4-RAG错；无Hyde错；空行；
# 题4 A: RAG对; RAW对; GPT4对; Reranker-RAG对；openai_xiaobei错；GPT4-RAG对；无Hyde对；空行；
# 题5 E: RAG对; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei错；GPT4-RAG对；无Hyde错；空行；
# 题7 A: RAG对; RAW对; GPT4对; Reranker-RAG对；openai_xiaobei对；GPT4-RAG错；无Hyde对；空行；
# 题8 C: RAG对; RAW错; GPT4错; Reranker-RAG对；openai_xiaobei错；GPT4-RAG错；无Hyde错；空行；
# 题9 B: RAG对; RAW错; GPT4对; Reranker-RAG错；openai_xiaobei错；GPT4-RAG对；无Hyde错；空行；
# 题10C: RAG对; RAW错; GPT4对; Reranker-RAG错；openai_xiaobei对；GPT4-RAG对；无Hyde对；空行；
# 题11C: RAG对; RAW错; GPT4对; Reranker-RAG错；openai_xiaobei错；GPT4-RAG错；无Hyde错；空行；
# 题12C: RAG错; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei对；GPT4-RAG错；无Hyde对；空行；
# 题13C: RAG错; RAW错; GPT4错; Reranker-RAG错；openai_xiaobei错；GPT4-RAG对；无Hyde错；空行；
# 题14D: RAG对; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei错；GPT4-RAG对；无Hyde对；空行；
# 题15A: RAG错; RAW错; GPT4错; Reranker-RAG对；openai_xiaobei对；GPT4-RAG对；无Hyde错；空行；
# 题16A: RAG对; RAW错; GPT4对; Reranker-RAG对；openai_xiaobei错；GPT4-RAG对；无Hyde对；空行；
# 题17C: RAG对; RAW对; GPT4错; Reranker-RAG对；openai_xiaobei错；GPT4-RAG错；无Hyde错；空行；



# GPT4 TEST
# 30C；GPT4对；QE-RAG对；
# 31C；GPT4对；QE-RAG对；
# 32C；GPT4对；QE-RAG对；
# 30C；GPT4；QE-RAG；
# 30C；GPT4；QE-RAG；
# 30C；GPT4；QE-RAG；
# 30C；GPT4；QE-RAG；
# 30C；GPT4；QE-RAG；

print("Problem:", title_idx)

# while True:
import json

data = '下列有关鳃弓的描述，错误的是：A: 由间充质增生形成 B: 人胚第4周出现, C: 相邻鳃弓之间为鳃沟, D: 共5对鳃弓 E: 位于头部两侧'
data = '以下是中国医师考试中规培结业考试的一道耳鼻咽喉科相关的单项选择题，请分析每个选项，并最后给出答案。\n甲状舌管囊肿及瘘管的特征性临床表现是\nA. 病变位于颈前正中，随吞咽上下活动\nB. 透光试验阳性\nC. 常与皮肤粘连\nD. 无痛性包块\nE. 病变部位摸到条索样物'
data = '以下是中国医师考试中规培结业考试的一道康复医学科相关的单项选择题，请分析每个选项，并最后给出答案。\n急性期脑卒中患者并发症的预防不包括\nA. 使用翻身床、气垫床等预防压疮\nB. 按摩促进血液、淋巴回流，减轻水肿，防止深静脉血栓形成\nC. 床上被动运动尽快过渡到主动活动，防止呼吸道感染、泌尿道感染\nD. 床上被动运动维持肌张力和关节活动度，预防关节挛缩变形\nE. 摇高床头半卧位，预防肢体肌肉痉挛'
data = '以下是中国医师考试中规培结业考试的一道眼科相关的多项选择题，请分析每个选项，并最后给出答案。\n结核性葡萄膜炎的眼底表现包括\nA. 视网膜血管炎\nB. 黄斑水肿\nC. 脉络膜炎\nD. 下方玻璃体的雪球样混浊\nE. 肉芽肿性前葡萄膜炎'
data = '以下是中国医师考试中规培结业考试的一道口腔科相关的单项选择题，请分析每个选项，并最后给出答案。\n选择拐杖长度，下列哪项不正确（）\nA. 患者身高减去40cm\nB. 患者下肢长度加上40cm\nC. 拐杖顶垫间相距约2～3m\nD. 拐杖底端应该侧离足跟15～20cm\n'
print(data)
data_dict = {'clear': 0, 'query': data}
client.send(data_dict)
data = client.recv()  # 等待接受数据
print(data)