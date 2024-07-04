import os
import sys
import torch
import warnings
import configparser
from typing import Tuple
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoModelForAudioClassification, Wav2Vec2FeatureExtractor)


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
            self.ate_model_id = self.config['AUDIOEMOTIONS']['ModelId']
            (self.ate_model,
             self.ate_feature_extractor,
             self.ate_sampling_rate) = self.__init_models(self.ate_model_id)

            self.__initialized = True

    def __get_config(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'audioConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'audioConfig.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self, ate_model_id) -> Tuple:
        # Audio to emotions
        ate_model = AutoModelForAudioClassification.from_pretrained(ate_model_id)
        ate_model.to(self.device)
        ate_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ate_model_id)
        ate_sampling_rate = ate_feature_extractor.sampling_rate

        return (ate_model,
                ate_feature_extractor,
                ate_sampling_rate)
