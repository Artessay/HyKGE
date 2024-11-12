import json
import requests
from datetime import datetime, timedelta

from .secret import secret_id, signature

PROXY_NUM = 1
PROXY_UPDATE_TIME = timedelta(minutes=4)

class ProxyManager:
    def __init__(self):
        self.api_url = ("https://dps.kdlapi.com/api/getdps?secret_id=" +
                        secret_id + "&signature=" + signature + "&num=" + str(PROXY_NUM) + '&format=json&sep=2')

        self.proxies = []   # 目前拥有的可用代理
        self.proxy_num = 0  # 目前拥有的可用代理数量
        self.proxy_idx = 0  
        self.last_updated = None # 上次更新时间

        self._update_proxies()  # 立即更新一次proxy

    def get_next_proxy(self):
        assert self.proxy_num > 0

        # 检查上一次更新时间是否超过限定时间
        if self.last_updated is None or (datetime.now() - self.last_updated) > PROXY_UPDATE_TIME:
            self._update_proxies()  # 超过，重新调用update方法更新proxy
        
        proxy = self.proxies[self.proxy_idx]
        self.proxy_idx = self.proxy_idx + 1 if self.proxy_idx != self.proxy_num - 1 else 0
        
        return proxy

    def _update_proxies(self):
        self.last_updated = datetime.now()  # 更新时间

        try:
            # 获取API接口返回的代理IP
            proxy_ip = requests.get(self.api_url).text
            
            proxy_ip = json.loads(proxy_ip)
            self.proxies = proxy_ip['data']['proxy_list']
        except Exception:
            if proxy_ip:
                try:
                    print("Error:", proxy_ip['msg']) 
                except:
                    print(proxy_ip)
            self.proxies = []

        self.proxy_num = len(self.proxies)
        self.proxy_idx = 0

        print(self.last_updated, "request proxies number:", self.proxy_num)

# 测试代码
if __name__ == '__main__':
    ip_manager = ProxyManager(num=1)
    proxy1 = ip_manager.get_next_proxy()  # 第一次显示，会更新IP地址
    print(proxy1)

    # 假设过了一段时间
    import time
    time.sleep(5)  # 等待5秒

    proxy2 = ip_manager.get_next_proxy()  # 没有超过4分钟，不会更新IP地址
    print(proxy2)

    assert proxy1 == proxy2

    # 假设过了更长时间
    time.sleep(250)  # 等待250秒（超过4分钟）

    proxy3 = ip_manager.get_next_proxy()  # 超过4分钟，会更新IP地址
    print(proxy3)

    assert proxy2 != proxy3

    time.sleep(10)  # 等待5秒

    proxy4 = ip_manager.get_next_proxy()  # 没有超过4分钟，不会更新IP地址
    print(proxy4)

    assert proxy3 == proxy4
