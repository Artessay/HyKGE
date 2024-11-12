#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Xinke Jiang
# @Email: XinkeJiang@stu.pku.edu.cn
# @Time: 2023/11/6 23:23
# @File: cut_stop_words.py

import re
import jieba
import numpy as np

def load_stop_words():
    # 读取停用词库
    with open('./utils/stop_words_file.txt', encoding='utf-8') as f:
        con = f.readlines()
        stop_words = set()
        for i in con:
            i = i.replace("\n", "")
            stop_words.add(i)
    return stop_words


# 定义分词函数
def chinese_word_cut(mytext, stop_words):

    # 文本预处理 ：去除一些无用的字符只提取出中文出来
    new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
    new_data = " ".join(new_data)

    # 文本分词
    seg_list_exact = jieba.lcut(new_data)
    result_list = []

    result_list = ''
    # 去除停用词并且去除单字
    for word in seg_list_exact:
        if word not in stop_words and len(word) > 1:
            result_list += word
            # result_list += ','
    return result_list

def chunk_text(text,chunk_size,overlap):
    text_array = np.array(list(text))
    # Calculate step and length for the strided array
    step = chunk_size - overlap
    length = (len(text) - chunk_size) // step + 1

    # Create a strided array of characters
    strided = np.lib.stride_tricks.as_strided(text_array, shape=(length, chunk_size), strides=(step*text_array.itemsize, text_array.itemsize))

    # Convert the strided array back to text
    chunks = [''.join(row) for row in strided]
    remaining_text = text[length * step+overlap:]
    if remaining_text:
        if chunks:
            chunks[-1] += remaining_text
        else:
            chunks.append(remaining_text)
    print(chunks)
    return chunks