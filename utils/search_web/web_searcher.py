import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from multiprocessing.dummy import Pool as ThreadPool

from .secret import user_id, user_secret
from .proxy_manager import ProxyManager

class WebSearcher(object):
    def __init__(self):
        self.UA = UserAgent()
        self.sess = requests.Session()
        self.sess_flag = True
        self.deafult_title = '页面没找到！'
        self.deafult_text = '这是一个消歧义页'
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.8",
            "Cache-Control": "max-age=0",
            "Connection": "close", # "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
        }

        self.proxy_manager = ProxyManager()

    def search_entity_description(self, query: str):
        try:
            url = f'http://www.a-hospital.com/w/{query}'

            proxy = self.proxy_manager.get_next_proxy()
            print(proxy)

            proxy = "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": user_id, "pwd": user_secret, "proxy": proxy}
            
            self.headers["User-Agent"] = self.UA.random
            response = requests.get(url=url, headers=self.headers, proxies={'http': proxy}, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                if soup.find('title').text == self.deafult_title or soup.find('p') is None or soup.find('p').text is None or self.deafult_text in soup.find('p').text:            # p 就是abstruct
                    return None
                else:
                    description = soup.find('p').text
                    return description
            else:
                return None
            
        except Exception as e:
            print(f'Error: {e}')
            return None
    
    '''
    search entity description for each entity name in the entity name list,
    return a dictionary that map title to its detail
    '''
    def search_entity_descriptions(self, entity_name_list: list):
        # search each entity and get description for them
        pool = ThreadPool()
        result_list = pool.map(self.search_entity_description, entity_name_list)
        pool.close()
        pool.join()

        entity_description = {}
        for (entity_name, result) in zip(entity_name_list, result_list):
            if result is not None:
                entity_description[entity_name] = result
   
        return entity_description