import os
import sys
import torch
import warnings
import configparser
from typing import Tuple
warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Pipeline as AudioPipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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
            self.diarization_model_id = self.config['DIARIZATION']['ModelId']
            self.stt_model_id = self.config['SPEECHTOTEXT']['ModelId']
            (self.diarization_pipeline,
             self.stt_model,
             self.stt_processor) = self.__init_models(self.diarization_model_id, self.stt_model_id)

            self.__initialized = True

    def __get_config(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'config.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'config.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self, diarization_model_id, stt_model_id) -> Tuple:
        # Diarization
        diarization_pipeline = AudioPipeline.from_pretrained(diarization_model_id,
                                                             use_auth_token=os.environ.get('HUGGINGFACE_TOKEN'))

        # Speech to text
        stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_id)
        stt_model.to(self.device)
        stt_processor = AutoProcessor.from_pretrained(stt_model_id)

        return (diarization_pipeline,
                stt_model,
                stt_processor)
