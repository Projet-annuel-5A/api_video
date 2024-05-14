import os
import sys
import torch
import shutil
import logging
import warnings
import configparser
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Pipeline as AudioPipeline
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSpeechSeq2Seq,
                          AutoProcessor,
                          AutoModelForAudioClassification,
                          Wav2Vec2FeatureExtractor,
                          AutoImageProcessor,
                          AutoModelForImageClassification)


# Create custom stream handler
class LoggerWriter:
    def __init__(self, logger, level) -> None:
        self.level = level
        self.logger = logger

    def write(self, message: str) -> None:
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self) -> None:
        pass


class BufferingHandler(logging.Handler):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.buffer = []
        self.filename = filename

    def emit(self, record: logging.LogRecord) -> None:
        # Append the log record to the buffer
        self.buffer.append(self.format(record))

    def flush(self) -> None:
        # Write the buffered log messages to the desired output
        if len(self.buffer) > 0:
            with open(self.filename, 'w') as f:
                for message in self.buffer:
                    f.write(message + '\n')


class Utils:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, session_id: str = None, interview_id: str = None, current_speaker: str = None) -> None:
        if not self.__initialized:

            self.config = self.__get_config()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.current_speaker = current_speaker

            # Folders
            base_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            base_folder = os.path.join(base_folder, self.config['FOLDERS']['Main'], session_id, interview_id)
            self.input_folder = os.path.join(base_folder, self.config['FOLDERS']['Input'])
            self.output_folder = os.path.join(base_folder, self.config['FOLDERS']['Output'])
            self.log_folder = os.path.join(self.output_folder, current_speaker, 'logs')
            self.temp_folder = os.path.join(self.output_folder, current_speaker, 'temp')
            self.output_audio_folder = os.path.join(self.output_folder, current_speaker, 'audio_parts')
            self.output_results_folder = os.path.join(self.output_folder, current_speaker, 'results')

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False
            sys.stderr = LoggerWriter(self.log, logging.ERROR)

            self.check_dirs()

            (self.diarization_pipeline,
             self.stt_model,
             self.stt_processor,
             self.tte_tokenizer,
             self.tte_model,
             #self.ate_model,
             #self.ate_feature_extractor,
             #self.ate_sampling_rate,
             self.vte_model,
             self.vte_processor
             ) = self.__init_models()

            self.__initialized = True

    def __create_folder(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            self.log.info("The directory {} has been created".format(path))
        else:
            self.log.info("The directory {} exists".format(path))

    def __init_logs(self) -> logging.Logger:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.NOTSET)

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        # Configure basic logging settings
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
        date_format = '%d/%b/%Y %H:%M:%S'
        encoding = 'utf-8'

        # Create a file handler for INFO messages
        info_log = os.path.join(self.log_folder, 'log_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        info_handler = BufferingHandler(info_log)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)

        # Create a file handler for ERROR messages
        error_log = os.path.join(self.log_folder, 'errorLog_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        error_handler = BufferingHandler(error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        root_logger.handlers.clear()

        # Add the handlers to the root logger
        root_logger.addHandler(info_handler)
        root_logger.addHandler(error_handler)
        root_logger.datefmt = date_format
        root_logger.encoding = encoding
        return root_logger

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

    def __init_models(self) -> tuple:
        # Diarization
        diarization_model_id = self.config['DIARIZATION']['ModelId']
        diarization_pipeline = AudioPipeline.from_pretrained(diarization_model_id,
                                                             use_auth_token=os.environ.get('HUGGINGFACE_TOKEN'))
        self.log.info('Diarization pipeline created with model {} from HuggingFace'.format(diarization_model_id))

        # Speech to text
        stt_model_id = self.config['SPEECHTOTEXT']['ModelId']
        stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_id)
        stt_model.to(self.device)
        stt_processor = AutoProcessor.from_pretrained(stt_model_id)

        # Text to emotions
        tte_model_id = self.config['TEXTEMOTIONS']['ModelId']
        tte_tokenizer = AutoTokenizer.from_pretrained(tte_model_id)
        tte_model = AutoModelForSequenceClassification.from_pretrained(tte_model_id)
        tte_model.to(self.device)
        self.log.info('Text-to-emotions model {} and tokenizer loaded in {}'.format(tte_model_id, self.device))

        '''
        # Audio to emotions
        ate_model_id = self.config['AUDIOEMOTIONS']['ModelId']
        ate_model = AutoModelForAudioClassification.from_pretrained(ate_model_id)
        ate_model.to(self.device)
        ate_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ate_model_id)
        ate_sampling_rate = ate_feature_extractor.sampling_rate
        self.log.info('Audio-to-emotions model {} loaded in {}'.format(ate_model_id, self.device))
        '''
        # Video to emotions
        vte_model_id = self.config['VIDEOEMOTION']['ModelId']
        vte_model = AutoModelForImageClassification.from_pretrained(vte_model_id, output_attentions=True)
        vte_model.to(self.device)
        vte_processor = AutoImageProcessor.from_pretrained(vte_model_id, output_attentions=True)
        self.log.info('Video-to-emotions model {} loaded in {}'.format(vte_model_id, self.device))

        return (diarization_pipeline,
                stt_model,
                stt_processor,
                tte_tokenizer,
                tte_model,
                #ate_model,
                #ate_feature_extractor,
                #ate_sampling_rate,
                vte_model,
                vte_processor)

    def check_dirs(self) -> None:
        if not os.path.exists(self.input_folder):
            message = "The folder '{}' does not exist, the program can not continue".format(self.input_folder)
            print(message)
            self.log.error(message)
            self.end_logs()
            sys.exit()
        else:
            self.log.info("The directory '{}' exists".format(self.input_folder))

        self.__create_folder(self.output_folder)
        self.__create_folder(self.output_audio_folder)
        self.__create_folder(self.output_results_folder)
        self.__create_folder(self.temp_folder)

    def delete_temp_files(self) -> None:
        shutil.rmtree(self.temp_folder)
        self.log.info('Temporary files deleted')

    def end_logs(self) -> None:
        log_handlers = logging.getLogger().handlers[:]
        for handler in log_handlers:
            handler.flush()
            logging.getLogger().removeHandler(handler)
