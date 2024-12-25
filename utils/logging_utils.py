import logging
import os

class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self,**kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(kwargs["log_path"])
        return self._instance[self._cls]

@Singleton
class MyLogger(object):
    def __init__(self,log_path=None):
        logging.basicConfig(level = logging.INFO, format = '[%(asctime)s] - %(levelname)s: %(message)s', filename=log_path, filemode='a')

    def info(self,msg):
        logging.info(msg)

    def warn(self,msg):
        logging.warn(msg)

    def debug(self,msg):
        logging.debug(msg)

    def error(self,msg):
        logging.error(msg)