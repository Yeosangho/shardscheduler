import os
import logging
import logging.handlers
from time import gmtime, strftime
logging_rank_list = [0]
class customlogging:
    __logger = logging.getLogger('SnowLog')
    __logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s > [%(levelname)s] %(message)s')
    slurm_job_id = os.environ.get("SLURM_JOB_ID", strftime("%Y-%m-%d-%H:%M:%S", gmtime()))
    
    fileHandler = logging.FileHandler(f'logs/{slurm_job_id}_train_log.txt')
    fileHandler.setFormatter(formatter)

    __logger.addHandler(fileHandler)


    @classmethod
    def debug(cls, rank, message):
        if(rank in logging_rank_list):
            cls.__logger.debug(f"{rank} :: {message}")

    @classmethod
    def info(cls, rank, message):
        if(rank in logging_rank_list):
            cls.__logger.info(f"{rank} :: {message}")

    @classmethod
    def warning(cls, rank, message):
        if(rank in logging_rank_list):
            cls.__logger.warning(f"{rank} :: {message}")        

    @classmethod
    def error(cls, rank, message):
        if(rank in logging_rank_list):
            cls.__logger.error(f"{rank} :: {message}")        

    @classmethod
    def critical(cls, rank, message):
        if(rank in logging_rank_list):
            cls.__logger.critical(f"{rank} :: {message}")        
