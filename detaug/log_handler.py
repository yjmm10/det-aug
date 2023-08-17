import os
from datetime import datetime
from loguru import logger
from pathlib import Path
import threading
import sys
import time

my_log_file_path =r'log.log'
class Log:
    # __instance = None
    _lock = threading.Lock()

    # 单例
    def __new__(cls, *args, **kwargs):
        time.sleep(1)
        with cls._lock:
            if not hasattr(cls, '__instance'):
            # if not cls.__instance:
                cls.__instance = super(Log, cls).__new__(cls, *args, **kwargs)
        return cls.__instance
    
    def __init__(self):
        # # 文件名称，按天创建
        DATE = datetime.now().strftime('%Y-%m-%d')
    
        # # 项目路径下创建log目录保存日志文件
        logpath = Path(os.getcwd()).joinpath("logs")  # 拼接指定路径
        logpath.mkdir(parents=True,exist_ok=True)
    
        logger.add('%s\%s_error.log' % (logpath, DATE), 
                level="ERROR",
                encoding='utf-8',   
                retention='1 days',  # 设置历史保留时长
                backtrace=True,  # 回溯
                diagnose=True,   # 诊断
                enqueue=True,   # 异步写入
                filter=lambda x: 'ERROR' in str(x['level']).upper())
        logger.add('%s\%s.log' % (logpath, DATE), 
                level="DEBUG",
                encoding='utf-8',   
                retention='1 days',  # 设置历史保留时长
                backtrace=True,  # 回溯
                diagnose=True,   # 诊断
                enqueue=True,   # 异步写入
                )

 
    def info(self, msg, *args, **kwargs):
        return logger.info(msg, *args, **kwargs)
 
    def debug(self, msg, *args, **kwargs):
        return logger.debug(msg, *args, **kwargs)
 
    def warning(self, msg, *args, **kwargs):
        return logger.warning(msg, *args, **kwargs)
 
    def error(self, msg, *args, **kwargs):
        return logger.error(msg, *args, **kwargs)
 
    def exception(self, msg, *args, exc_info=True, **kwargs):
        return logger.exception(msg, *args, exc_info=True, **kwargs)

       
        

if __name__ == '__main__':
    log = Log()
    log.info("im vip")
    log.debug("这是个debug")
    log.warning("这是个warning")
    log.error("这是个error")