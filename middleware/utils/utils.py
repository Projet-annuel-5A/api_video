import os
import sys
import torch
import shutil
import logging
import warnings
import configparser
from datetime import datetime
from typing import Tuple, Any
from dotenv import load_dotenv
from supabase import create_client, Client
warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Pipeline as AudioPipeline
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor)


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
            load_dotenv()
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

            self.supabase_client = self.__check_supabase_connection()
            self.supabase_connection = self.__connect_to_bucket()

            self.check_dirs()

            self.__download_input_file(session_id, interview_id)

            (self.diarization_pipeline,
             self.stt_model,
             self.stt_processor) = self.__init_models()

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

    def __init_models(self) -> Tuple:
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

        return (diarization_pipeline,
                stt_model,
                stt_processor)

    def check_dirs(self) -> None:
        self.__create_folder(self.input_folder)
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

    def __check_supabase_connection(self) -> Client:
        try:
            client = create_client(self.config['SUPABASE']['Url'], os.environ.get('SUPABASE_KEY'))
        except Exception as e:
            message = ('Error connecting to Supabase, the program can not continue. {}'.
                       format(e.args[0]['message']))
            self.log.error(message)
            print(e)
            sys.exit(1)
        return client

    def __connect_to_bucket(self) -> Any:
        bucket_name = self.config['SUPABASE']['InputBucket']
        connection = self.supabase_client.storage.from_(bucket_name)
        try:
            connection.list()
            self.log.info('Connection to S3 bucket {} successful'.format(bucket_name))
        except Exception as e:
            message = ('Error connecting to S3 bucket {}, the program can not continue. {}'.
                       format(bucket_name, e.args[0]['message']))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return connection

    def __download_input_file(self, session_id: str, interview_id: str) -> None:
        videoname = self.config['GENERAL']['Filename']
        s3_path = '{}/{}/raw/{}'.format(session_id, interview_id, videoname)
        try:
            with open(os.path.join(self.input_folder, videoname), 'wb+') as f:
                res = self.supabase_connection.download(s3_path)
                f.write(res)
            self.log.info('The file {} has been downloaded from the S3 bucket'.format(videoname))
        except Exception as e:
            message = ('Error downloading the file {} from the S3 bucket: {}'.
                       format(videoname, e.args[0]['message']))
            self.log.error(message)
            print(message)
            sys.exit(1)
