import json
import os
from multiprocessing.dummy import Pool as ThreadPool

from . import WebSearcher

class EncyclopediaDictionary:
    def __init__(self, cache_file='dictionary_cache.json'):
        self.cache_file = cache_file
        self.cache_dirty = False

        self.cache = self.load_cache()
        self.web_searcher = WebSearcher()
        
    def load_cache(self):
        # 尝试从文件中加载缓存
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_cache(self):
        # 将缓存保存到文件
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        
    def lookup_word(self, word):
        # 检查单词是否在缓存中
        if word in self.cache:
            return self.cache[word]
        else:
            # 从网络查找
            meaning = self.web_searcher.search_entity_description(word)
            # 将结果存入缓存
            self.cache[word] = meaning
            # 设置dirty bit
            self.cache_dirty = True
            return meaning
    
    def search_entity_descriptions(self, word_list: list):
        # search each entity and get description for them
        pool = ThreadPool()
        result_list = pool.map(self.lookup_word, word_list)
        pool.close()
        pool.join()

        # 保存缓存
        if self.cache_dirty:
            self.save_cache()
            self.cache_dirty = False

        entity_description = {}
        for (entity_name, result) in zip(word_list, result_list):
            if result is not None:
                entity_description[entity_name] = result
   
        return entity_description

# 使用示例
if __name__ == "__main__":
    dictionary = EncyclopediaDictionary()
    
    # 查询单词
    word = 'hello'
    print(dictionary.lookup_word(word))  # 查询并可能缓存结果
    print(dictionary.lookup_word(word))  # 第二次查询，从缓存中获取

    # 程序结束前，确保缓存被保存
    dictionary.save_cache()
