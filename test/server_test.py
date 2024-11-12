import io
import pickle
import sys
import threading
from multiprocessing.connection import Client
import threading
import time
from socket import *
import json

def send_data(data):
    # 创建socket连接
    client = Client(('0.0.0.0', 8205))
    
    # 构建要发送的数据字典
    data_dict = {'clear': 0, 'query': data}
    
    # 发送数据
    client.send(data_dict)
    res = client.recv()  # 等待接受数据
    print(res)
    client.close()

# 要发送的数据
data_1 = "你好，我最近有点口腔溃疡，可以吃什么药？"
data_2 = "我发烧了怎么办？"
data_3 = "我肚子疼怎么办？"

# # 创建线程
# thread1 = threading.Thread(target=send_data, args=(data_1,))


# # 启动线程
# thread1.start()


# # 等待线程执行结束
# thread1.join()

# print("发送完毕")
thread2 = threading.Thread(target=send_data, args=(data_2,))
thread3 = threading.Thread(target=send_data, args=(data_3,))
thread2.start()
thread3.start()
thread2.join()
thread3.join()
print("发送完毕")