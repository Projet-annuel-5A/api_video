import os
import sys
import torch
import logging
import warnings
import configparser
from typing import Tuple
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)


class Utils:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, session_id: str, interview_id: str, current_speaker: str) -> None:
        if not self.__initialized:

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.config = self.__get_config()
            # Folders
            base_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_folder = os.path.join(base_folder, self.config['FOLDERS']['Main'], session_id, interview_id)
            self.output_folder = os.path.join(base_folder, self.config['FOLDERS']['Output'])
            self.log_folder = os.path.join(self.output_folder, current_speaker, 'logs')

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False

            (self.tte_tokenizer,
             self.tte_model) = self.__init_models()

            self.__initialized = True

    def __init_logs(self) -> logging.Logger:
        logger = logging.getLogger('text')
        logger.setLevel(logging.NOTSET)

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        # Configure basic logging settings
        formatter = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        date_format = '%d/%b/%Y %H:%M:%S'
        encoding = 'utf-8'

        # Create a file handler for INFO messages
        info_text_log = os.path.join(self.log_folder, 'logText_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        info_text_handler = logging.FileHandler(info_text_log)
        info_text_handler.setLevel(logging.INFO)
        info_text_handler.setFormatter(logging.Formatter(formatter))

        logger.handlers.clear()

        # Add the handlers to the root logger
        logger.addHandler(info_text_handler)
        logger.datefmt = date_format
        logger.encoding = encoding
        return logger

    def __get_config(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'textConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'textConfig.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self) -> Tuple:
        # Text to emotions
        tte_model_id = self.config['TEXTEMOTIONS']['ModelId']
        tte_tokenizer = AutoTokenizer.from_pretrained(tte_model_id)
        tte_model = AutoModelForSequenceClassification.from_pretrained(tte_model_id)
        tte_model.to(self.device)
        self.log.info('Text-to-emotions model {} and tokenizer loaded in {}'.format(tte_model_id, self.device))

        return (tte_tokenizer,
                tte_model)
