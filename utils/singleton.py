import threading


def Singleton(cls):
    # print("log singleton")
    cls._instance = None
    cls._lock = threading.Lock()

    def __new__(*args, **kwargs): 
        # print(f"log new", args, kwargs)     # 实际上args中也包含cls和相关的参数
        with cls._lock:
            if not cls._instance:
                cls._instance = object.__new__(cls)
        return cls._instance

    cls.__new__ = __new__
    return cls
