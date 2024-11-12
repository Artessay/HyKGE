from module import HOModule,NERModule,KGModule,DOCModule,FilterModule,KOModule
from config import Config
from utils.singleton import Singleton



@Singleton
class ModuleManager:
    def __init__(self):
        pass

    def setup(self,config: Config = None):
        self.HOModule = HOModule(config)
        self.NERModule = NERModule(config)
        self.KGModule = KGModule(config)
        self.DOCModule = DOCModule(config)
        self.FilterModule = FilterModule(config)
        self.KOModule = KOModule(config)

    def mapper(self):
        print("in class")
        print(self)
        return self.HOModule,self.NERModule,self.KGModule,self.DOCModule,self.FilterModule,self.KOModule
    





