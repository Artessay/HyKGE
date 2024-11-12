from config import Config
from utils.prompt_build import *
from openai import OpenAI
from utils.history_qa import *
from utils.singleton import Singleton
from dashscope import Generation
import os

@Singleton
class KOModule:
    def __init__(self,config: Config = None):
        self.HyDE = config.configHO.HyDE
        self.query_expansion = config.configHO.query_expansion
        self.LLM_type = config.configHO.LLM_type
        self.hyde_max_token_num = config.configHO.hyde_max_token_num
        self.history_epoch = config.configServer.history_epoch
        # LLM_type_list = ['GPT4', 'GPT3.5', 'Xiaobei', 'Baichuan2-Chat', 'Huatuo', 'QWen']
        if self.LLM_type == 'GPT3.5' or self.LLM_type == 'GPT4':
            self.openai_client = OpenAI(
            # 输入转发API Key
            api_key=os.environ['OPENAI_API_KEY'],
            base_url="https://api.chatanywhere.com.cn/v1"
        )
        if self.LLM_type == 'QWen':
            assert False
            self.openai_client = OpenAI(api_key="xxxx", base_url="xxxx")
        if self.LLM_type == 'Ali':
            # 设置环境变量
            # 这是online的72B
            assert os.environ['DASHSCOPE_API_KEY'] is not None
            self.openai_client = Generation()


    def process(self, gold_triplets, entity_description, sentence_list, query):
        gold_triplet_pre = ""
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

        if 1==1:
            ko_prompt = build_KO_prompt(query, gold_triplet_pre, entity_description, sentence_list)
            print(ko_prompt)
            if "QWen" in self.LLM_type:
                # chat_response = self.openai_client.chat.completions.create(
                #     model="qwen/Qwen1.5-14B-Chat",
                #     messages=[
                #         {"role": "system", "content": "You are a helpful assistant."},
                #         {"role": "user", "content": ko_prompt},
                #     ],
                #     temperature=0.95,
                #     max_tokens=self.hyde_max_token_num,
                # )
                # ko_out = chat_response.choices[0].message.content.strip()

                chat_response = self.openai_client.chat.completions.create(
                    model="Qwen-72B-Chat-Int4",  # 下面列出来的任意模型名字
                    messages=[{"role": "user", "content": ko_prompt}])
                ko_out = chat_response.choices[0].message.content

            elif 'Ali' in self.LLM_type:
                chat_response = self.openai_client.call(
                    Generation.Models.qwen_max,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": ko_prompt},
                    ],
                    result_format='message',  # set the result to be "message" format.
                    max_tokens=self.hyde_max_token_num,
                )
                ko_out = chat_response.output.choices[0].message.content.strip()

            elif 'GPT' in self.LLM_type:
                if self.LLM_type == 'GPT4':
                    model = "gpt-4-vision-preview"
                if self.LLM_type == 'GPT3.5':
                    model = "gpt-3.5-turbo-1106"

                ko_out = ""
                try:
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": ko_prompt},
                                ],
                            }
                        ],
                        max_tokens=self.hyde_max_token_num,
                        stream=False  # # 是否开启流式输出
                    )
                    ko_out = response.choices[0].message.content  # Should be include in Try block
                    # print(response.choices[0].text.strip())
                except Exception as e:
                    print(f"Openai Connect Error: {e}")

            else:
                ko_out = talk_to_LLM(ko_prompt, response_num=1, max_tokens=self.hyde_max_token_num)

    #             hyde_data="""
    # 头痛的治疗方法一般可以分为急性治疗和预防治疗两种。在急性治疗方面，常用的药物包括非甾体抗炎药（如布洛芬、阿司匹林、对乙酰氨基酚）、三环类抗抑郁药（如阿米替林）、三环类抗抑郁药与对血管活性的β受体拮抗药的联合制剂（如曲马多）、以及特定的镇痛药（如曲唑酮）。而在预防治疗方面，常用的药物包括β受体阻滞剂、血管紧张素转换酶抑制剂、钙通道阻滞剂和抗抑郁药等
    #  """

        return ko_out