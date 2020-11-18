# -*- coding: utf-8 -*-
# @Time     : 2020/11/16 20:13
# @Author   : lishijie
import logging

def setup_logger(log_file_path: str = None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('hyperIQA')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    
    return logger