import os
import sys
import torch
import warnings
import configparser
from typing import Tuple
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)


class Models:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self.__initialized:
            self.config = self.__get_config()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tte_model_id = self.config['TEXTEMOTIONS']['ModelId']
            (self.tte_tokenizer,
             self.tte_model) = self.__init_models(self.tte_model_id)

            self.__initialized = True

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

    def __init_models(self, tte_model_id) -> Tuple:
        # Text to emotions
        tte_tokenizer = AutoTokenizer.from_pretrained(tte_model_id)
        tte_model = AutoModelForSequenceClassification.from_pretrained(tte_model_id)
        tte_model.to(self.device)

        return (tte_tokenizer,
                tte_model)
