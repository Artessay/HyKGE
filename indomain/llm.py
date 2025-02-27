from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_aws import ChatBedrock
from botocore.config import Config
from openai import OpenAI

# fixme your config

generation_config = Config(read_timeout=120)

llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview", 
    temperature=0.0, 
    max_tokens=None,
    azure_endpoint="https://dev-nk-test.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview",
    api_key="xxxx"
)

json_llm = llm.bind(response_format={"type": "json_object"})

# 定义一个函数，调用LLM模型生成响应内容
# Define a function to generate messages, ensuring the format is correct
def llm_generate(messages):
    try:
        # Ensure 'messages' contains 'json' or a valid structure for response_format='json_object'
        # We can add a check or modify the content if necessary

        # Example modification:
        for message in messages:
            if "json" not in message["content"]:
                message["content"] = "json: " + message["content"]  # Adding a 'json' prefix to content

        # Call the LLM API with updated messages
        response = json_llm.invoke(messages)
        
        return response.content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None


# 定义生成器类，用于处理查询和返回答案
class Generator:
    def generate(self, refined_query: str) -> str:
        """
        根据 refined_query 构建消息并调用 LLMClient.generate 获取最终答案。
        :param refined_query: 用户的查询问题
        :return: 返回LLM模型生成的答案
        """
        # 构造消息列表，包含系统角色和用户输入
        messages = [
            {"role": "system", "content": "You are a helpful assistant that uses provided subgraph and supporting facts to answer questions."},
            {"role": "user", "content": refined_query}
        ]
        
        # 通过调用llm_generate函数获取响应
        response = llm_generate(messages)
        
        # 返回生成的响应内容，如果发生错误则返回适当的消息
        return response if response else "Error generating response"

# client = OpenAI(api_key="msraadmin", base_url="http://")

# # 定义生成器类，用于处理查询和返回答案
# class Generator:
#     def generate(self, refined_query: str) -> str:
#         """
#         根据 refined_query 构建消息并调用 LLMClient.generate 获取最终答案。
#         :param refined_query: 用户的查询问题
#         :return: 返回LLM模型生成的答案
#         """
#         # 构造消息列表，包含系统角色和用户输入
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant that uses provided subgraph and supporting facts to answer questions."},
#             {"role": "user", "content": refined_query}
#         ]
        
#         # 通过调用llm_generate函数获取响应
#         response = client.chat.completions.create(
#             model="qwen2.5:72b",
#             messages=messages,
#             temperature=0.0,    
#         )

#         response = response.choices[0].message.content

#         # 返回生成的响应内容，如果发生错误则返回适当的消息
#         return response if response else "Error generating response"
    