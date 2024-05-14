import os
import sys
import torch
import logging
import warnings
import configparser
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoModelForAudioClassification,
                          Wav2Vec2FeatureExtractor)


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
            print('Folder: ', self.config['FOLDERS']['Main'])
            # Folders
            base_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_folder = os.path.join(base_folder, self.config['FOLDERS']['Main'], session_id, interview_id)
            self.output_folder = os.path.join(base_folder, self.config['FOLDERS']['Output'])
            self.log_folder = os.path.join(self.output_folder, current_speaker, 'logs')
            self.output_audio_folder = os.path.join(self.output_folder, current_speaker, 'audio_parts')

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False

            (self.ate_model,
             self.ate_feature_extractor,
             self.ate_sampling_rate
            ) = self.__init_models()

            self.__initialized = True

    def __init_logs(self) -> logging.Logger:
        logger = logging.getLogger('audio')
        logger.setLevel(logging.NOTSET)

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        # Configure basic logging settings
        formatter = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        date_format = '%d/%b/%Y %H:%M:%S'
        encoding = 'utf-8'

        # Create a file handler for INFO messages
        info_audio_log = os.path.join(self.log_folder, 'logAudio_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        info_audio_handler = logging.FileHandler(info_audio_log)
        info_audio_handler.setLevel(logging.INFO)
        info_audio_handler.setFormatter(logging.Formatter(formatter))


        logger.handlers.clear()

        # Add the handlers to the root logger
        logger.addHandler(info_audio_handler)
        logger.datefmt = date_format
        logger.encoding = encoding
        return logger

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

    def __init_models(self) -> tuple:
        # Audio to emotions
        ate_model_id = self.config['AUDIOEMOTIONS']['ModelId']
        ate_model = AutoModelForAudioClassification.from_pretrained(ate_model_id)
        ate_model.to(self.device)
        ate_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ate_model_id)
        ate_sampling_rate = ate_feature_extractor.sampling_rate
        self.log.info('Audio-to-emotions model {} loaded in {}'.format(ate_model_id, self.device))

        return (ate_model,
                ate_feature_extractor,
                ate_sampling_rate
                )