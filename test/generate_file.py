from openai import OpenAI
import json
import requests
import datetime
from multiprocessing.connection import Client

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
    # print("Answer is:", da["answer"])
    return query

def generate_file_framework(origin_file_path,target_file_dir,target_file_name,checkpoint=0):
    res_list = []
    final_file_path = target_file_dir+"/"+target_file_name
    parts = target_file_name.split('.')
    tmp_file_path = target_file_dir+"/"+parts[0] + "_before_{}" + "." + parts[1]

    if checkpoint%50 !=0:
        print("checkpoint is invalid!!!")
        return

    if checkpoint != 0:
        f = open(tmp_file_path.format(checkpoint), 'r')
        content = f.read()
        res_list = json.loads(content)
        f.close()

    f = open(origin_file_path, 'r')

    content = f.read()
    data_list = json.loads(content)
    ix = 0
    print(len(data_list))
    for item in data_list:
        print("{}/{}".format(ix, len(data_list)))
        if ix < checkpoint:
            ix += 1
            continue
        if ix % 50 == 0 and ix != checkpoint:
            json.dump(res_list, open(tmp_file_path.format(ix), 'w'),
                    ensure_ascii=False)
        starttime = datetime.datetime.now()
        # todo do your work here and merge it into item
        data = item["question"]
        #data = get_query(item)
        data_dict = {'clear': 0, 'query': data}
        client = Client(('0.0.0.0', 8205))
        client.send(data_dict)
        res = client.recv()  # 等待接受数据
        #res= client 交互
        #res = res +'\n最后请以：”答案是：xx选项“的方式输出结果。'
        #res = res +'请详细且高质量的生成回答。'
        item['RAG_prompt']= res
        print(res)
        
        # if "rag_knowledge" in item:
        #     rag_knowledge = item["rag_knowledge"]
        #     question = item["question"]
        #     option_str=item["option_str"]
        #     result = [s for s in rag_knowledge.split('\n') if s.strip() and len(s.strip()) > 1]
        #     if(len(result)>1):
        #         prompt = PROMPT_TEMPLATE.format(question=question+option_str,doc=result[1],kg=result[0])
        #         res = chat_completion(prompt)
        #         item["convert_knowledge"] = res
        #todo do your work before
        endtime = datetime.datetime.now()
        print("use time:", (endtime - starttime).seconds, "s")
        res_list.append(item)
        ix += 1

    json.dump(res_list, open(final_file_path, 'w'), ensure_ascii=False)

f = open("./cmb_test/CMB_generate_inner_answer_0528_before_34500.json", 'r')
content = f.read()
res_list = json.loads(content)
f.close()

json.dump(res_list[17950:34500], open("CMB_generate_inner_answer_0612.json", 'w'), ensure_ascii=False)
