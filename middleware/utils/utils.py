import os
import sys
import torch
import logging
import tempfile
import warnings
import numpy as np
import configparser
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydub import AudioSegment
from typing import Tuple, Any, List
from moviepy.editor import VideoFileClip
from supabase import create_client, Client
warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Pipeline as AudioPipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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

    def flush(self) -> str:
        if len(self.buffer) > 0:
            return '\n'.join(self.buffer)
        else:
            return ''
            # with open(self.filename, 'w') as f:
            #    for message in self.buffer:
            #        f.write(message + '\n')


class Utils:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, session_id: int = None, interview_id: int = None) -> None:
        if not self.__initialized:
            load_dotenv()
            self.config = self.__get_config()

            self.session_id = session_id
            self.interview_id = interview_id
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # S3 Folders
            self.output_s3_folder = '{}/{}/output'.format(self.session_id, self.interview_id)

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False
            sys.stderr = LoggerWriter(self.log, logging.ERROR)

            self.supabase_client = self.__check_supabase_connection()
            self.supabase_connection = self.__connect_to_bucket()

            (self.diarization_pipeline,
             self.stt_model,
             self.stt_processor) = self.__init_models()

            self.__initialized = True

    def __init_logs(self) -> logging.Logger:
        root_logger = logging.getLogger('mainLog')
        root_logger.setLevel(logging.NOTSET)

        # Configure basic logging settings
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
        date_format = '%d/%b/%Y %H:%M:%S'
        encoding = 'utf-8'

        # Create a file handler for INFO messages
        info_log = 'log_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S'))
        info_handler = BufferingHandler(info_log)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)

        # Create a file handler for ERROR messages
        error_log = 'errorLog_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S'))
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

    def delete_temp_files(self, files: List[str]) -> None:
        for file in files:
            if os.path.exists(file):
                os.remove(file)
        s3_temp_path = '{}/temp'.format(self.output_s3_folder)
        for file in self.supabase_connection.list(s3_temp_path):
            filepath = '{}/{}'.format(s3_temp_path, file['name'])
            self.supabase_connection.remove(filepath)
        self.supabase_connection.remove(s3_temp_path)


    def end_logs(self) -> None:
        log_handlers = logging.getLogger('mainLog').handlers[:]
        for handler in log_handlers:
            if isinstance(handler, BufferingHandler):
                log = handler.flush()
                if log:
                    self.save_to_s3('{}.log'.format(handler.filename), log.encode(), 'text', 'logs')
            logging.getLogger('mainLog').removeHandler(handler)

    def __check_supabase_connection(self) -> Client:
        try:
            client = create_client(self.config['SUPABASE']['Url'], os.environ.get('SUPABASE_KEY'))
        except Exception as e:
            message = ('Error connecting to Supabase, the program can not continue.', str(e))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return client

    def __connect_to_bucket(self) -> Any:
        bucket_name = self.config['SUPABASE']['InputBucket']
        connection = self.supabase_client.storage.from_(bucket_name)
        try:
            connection.list()
            self.log.info('Connection to S3 bucket {} successful'.format(bucket_name))
        except Exception as e:
            message = ('Error connecting to S3 bucket {}, the program can not continue.'.
                       format(bucket_name), str(e))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return connection

    def save_to_s3(self, filename: str, content: Any, file_format: str, s3_subfolder: str = None) -> None:
        match file_format:
            case 'audio':
                content_type = 'audio/mpeg'
            case 'video':
                content_type = 'video/mp4'
            case 'text':
                content_type = 'text/plain'
            case _:
                content_type = 'text/plain'

        try:
            s3_path = '{}/{}/{}'.format(self.output_s3_folder,
                                        s3_subfolder,
                                        filename) if s3_subfolder else '{}/{}'.format(self.output_s3_folder, filename)
            self.supabase_connection.upload(file=content, path=s3_path, file_options={"content-type": content_type})
            self.log.info('File {} uploaded to S3 bucket at {}'.format(filename, s3_path))
        except Exception as e:
            message = ('Error uploading the file {} to the S3 bucket.'.
                       format(filename), str(e))
            self.log.error(message)

    '''
    def audiofileclip_to_tensor(self, audio_clip):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name

            try:
                # Export the AudioFileClip to the temporary file
                audio_clip.write_audiofile(temp_file_path, codec='pcm_s16le')

                # Read the temporary file with torchaudio
                waveform, sample_rate = torchaudio.load(temp_file_path)

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        return waveform, sample_rate
    '''

    def audiosegment_to_tensor(self, audio_segment: AudioSegment) -> torch.Tensor:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Extract raw audio data
        raw_data = np.array(audio_segment.get_array_of_samples())

        # Convert to NumPy array
        if audio_segment.channels == 2:  # Stereo audio
            raw_data = raw_data.reshape((-1, 2)).T.astype(np.float32)
        else:  # Mono audio
            raw_data = raw_data.astype(np.float32)

        # Normalize the audio data (optional)
        # Convert the raw data to range [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit audio
            raw_data /= 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit audio
            raw_data /= 2147483648.0

        # Convert to Torch Tensor
        audio_tensor = torch.from_numpy(raw_data)

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        return audio_tensor

    def merge_results(self) -> None:
        self.log.info('Merging results from text, audio and video processing')

        text_results = self.__read_df_from_s3('text_emotions')
        audio_results = self.__read_df_from_s3('audio_emotions')
        video_results = self.__read_df_from_s3('video_emotions')

        # Merge the results on 'speaker' and 'part'
        results = pd.merge(text_results, audio_results, on=['speaker', 'part'], how='inner')
        results = pd.merge(results, video_results, on=['speaker', 'part'], how='inner')

        # Save the results file to S3
        filename = 'results.h5'
        s3_path = '{}/results/{}'.format(self.output_s3_folder, filename)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                # Write the file to the temporary file
                with pd.HDFStore(temp_file_path) as store:
                    store.put('text', text_results)
                    store.put('video', video_results)
                    store.put('audio', audio_results)
                    store.put('all', results)

                with open(temp_file_path, 'rb') as f:
                    try:
                        self.supabase_connection.upload(file=f, path=s3_path,
                                                        file_options={'content-type': 'application/octet-stream'})
                        self.log.info('File {} uploaded to S3 bucket at {}'.format(filename, s3_path))
                    except Exception as e:
                        message = (
                            'Error uploading the file {} to the S3 bucket. '.format(filename), str(e))
                        self.log.info(message)
            finally:
                temp_file.close()
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        self.log.info('Results merged successfully')

    def open_input_file(self, videoname: str) -> Tuple[AudioSegment, str, str] | None:
        # if env == 'S3':
        s3_path = '{}/{}/raw/{}'.format(self.session_id, self.interview_id, videoname)
        try:
            video_bytes = self.supabase_connection.download(s3_path)
            self.log.info('Getting file {} from the S3 bucket'.format(videoname))
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file_path = temp_file.name
                try:
                    # Write the video_bytes to the temporary file
                    temp_file.write(video_bytes)
                    # Ensure data is written to disk
                    temp_file.flush()
                    self.log.info('Starting audio extraction from video file')
                    # Open the video from the temporary file and extract the audio
                    audio = VideoFileClip(temp_file_path).audio
                    self.log.info('Audio extraction finished')

                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file_2:
                        temp_file_path_2 = temp_file_2.name
                        try:
                            # Write the AudioFileClip to the temporary file
                            audio.write_audiofile(temp_file_path_2, codec='pcm_s16le')

                            # Load the temporary file with pydub
                            audio_segment = AudioSegment.from_file(temp_file_path_2, format='wav')
                        finally:
                            temp_file_2.close()
                finally:
                    temp_file.close()
        except Exception as e:
            message = ('Error downloading the file {} from the S3 bucket. '.
                       format(videoname), str(e))
            self.log.error(message)
            sys.exit(1)
        '''
        elif env == 'Local':
            filename = self.config['GENERAL']['Filename']
            base_folder = os.path.join(os.path.dirname(__file__),
                                       self.config['FOLDERS']['Main'],
                                       self.session_id,
                                       self.interview_id)
            video_path = os.path.join(base_folder, self.config['FOLDERS']['Input'], filename)
            self.log.info('File {} opened from local system'.format(videoname))
            video = VideoFileClip(video_path)
            self.log.info('Starting audio extraction from video file')
            audio = video.audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file_2:
                temp_file_path_2 = temp_file_2.name
                try:
                    # Write the AudioFileClip to the temporary file
                    audio.write_audiofile(temp_file_path_2, codec='pcm_s16le')

                    # Load the temporary file with pydub
                    audio_segment = AudioSegment.from_file(temp_file_path_2, format='wav')
                finally:
                    temp_file_2.close()
            self.log.info('Audio extraction finished')
            temp_file_path = filename
        else:
            return None
        '''
        return audio_segment, temp_file_path, temp_file_path_2

    def update_bool_db(self, interview_id: str, champ_name: str) -> None:
        self.log.info('Updating {} in the database'.format(champ_name))
        try:
            self.supabase_client.table('interviews').update({champ_name: True}).eq('id', interview_id)
            self.log.info('{} updated successfully'.format(champ_name))
        except Exception as e:
            message = ('Error updating {} in the database'.format(champ_name), str(e))
            self.log.error(message)

    def df_to_temp_s3(self, df: pd.DataFrame, filename: str) -> None:
        s3_path = '{}/temp/{}.tmp'.format(self.output_s3_folder, filename)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                df.to_hdf(temp_file_path, key='data', mode='w', complevel=9, complib='blosc')

                with open(temp_file_path, 'rb') as f:
                    try:
                        self.supabase_connection.upload(file=f, path=s3_path,
                                                        file_options={'content-type': 'application/octet-stream'})
                        self.log.info('File {} uploaded to S3 bucket'.format(s3_path))
                    except Exception as e:
                        message = (
                            'Error uploading the file to the S3 bucket. ', str(e))
                        self.log.info(message)
            finally:
                temp_file.close()
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    def __read_df_from_s3(self, filename) -> pd.DataFrame:
        path = '{}/temp/{}.tmp'.format(self.output_s3_folder, filename)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                res = self.supabase_connection.download(path)
                temp_file.write(res)
                df = pd.read_hdf(temp_file_path, key='data', index_col=None)
            finally:
                temp_file.close()
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        return df