import time
import json
import uvicorn
import requests
import threading
from tqdm import tqdm
from typing import Dict
from queue import Queue
from utils.utils import Utils
from dotenv import load_dotenv
from pydub import AudioSegment
from utils.diarizator import Diarizator
from utils.audioSplit import AudioSplit
from fastapi import FastAPI, HTTPException

app = FastAPI()


class Process:
    def __init__(self, session_id: int, interview_id: int):
        load_dotenv()
        self.session_id = session_id
        self.interview_id = interview_id
        self.utils = Utils(session_id, interview_id)

    def __analyze_text(self, params: Dict) -> bool:
        url = 'http://127.0.0.1:8002/analyse_text'
        response = requests.post(url, params=params)

        if response.status_code == 200:
            # TODO update value in DB
            return True
        else:
            self.utils.log.error("Error:", response.status_code)
            return False

    def __analyse_video(self, params: Dict) -> bool:
        url = 'http://127.0.0.1:8003/analyse_video'
        response = requests.post(url, params=params)

        if response.status_code == 200:
            # TODO update value in DB
            return True
        else:
            self.utils.log.error("Error:", response.status_code)
            return False

    def __analyse_audio(self, params: Dict) -> bool:
        url = 'http://127.0.0.1:8001/analyse_audio'
        response = requests.post(url, params=params)

        if response.status_code == 200:
            # TODO update value in DB
            return True
        else:
            self.utils.log.error("Error:", response.status_code)
            return False

    def __split_audio(self, _audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> None:
        asp = AudioSplit()
        self.utils.log.info('Starting split audio')
        texts = asp.process(_audiofile, _speakers, lang)
        # Save texts to S3
        self.utils.df_to_temp_s3(texts, filename='texts')
        self.utils.log.info('Split audio finished')
        # TODO update diarization champ in DB

    def __process_all(self, queue: Queue) -> None:
        drz = Diarizator()
        temp_files = []
        params = {
            'session_id': self.session_id,
            'interview_id': self.interview_id
        }

        try:
            # Create a queue to store the results
            filename = self.utils.config['GENERAL']['Filename']
            # Extract the audio from the video file
            audio_file, temp_file_path, temp_file_path_2 = self.utils.open_input_file(filename)
            temp_files.append(temp_file_path)
            temp_files.append(temp_file_path_2)
            audio_name = '{}.wav'.format(filename.split('.')[0])

            # Diarize and split the audio file
            speakers = drz.process(audio_file, audio_name)
            self.utils.save_to_s3('speakers.json', json.dumps(speakers).encode(), 'text', 'temp')
            self.__split_audio(audio_file, speakers, self.utils.config['GENERAL']['Language'])

            self.utils.log.info('Starting emotions detection from text, audio and video')
            text_results = self.__analyze_text(params)
            audio_results = self.__analyse_audio(params)
            video_results = self.__analyse_video(params)
            self.utils.log.info('Emotions detection threads from text, audio and video have finished')

            if text_results and video_results and audio_results:
                self.utils.merge_results()
            result = (True, None)
        except Exception as e:
            result = (False, e)
        finally:
            self.utils.delete_temp_files(temp_files)
        queue.put(result)

    def start_process(self):
        try:
            print('Program started')

            main_queue = Queue()

            self.utils.log.info("Program started => Session: {} | Interview: {}".format(self.session_id,
                                                                                        self.interview_id))

            # Start the process function in a separate thread
            main_thread = threading.Thread(target=self.__process_all, args=(main_queue, ))
            main_thread.start()

            # Create an indeterminate progress bar
            with tqdm(desc='Work in progress', unit='iterations', leave=True, disable=True) as progress_bar:
                i = 0
                while main_thread.is_alive():
                    progress_bar.write('\rProcessing... {}'.format('*' * i), end='', nolock=True)
                    i += 1
                    time.sleep(3)
                    progress_bar.refresh()

            # Wait for the main thread to finish
            main_thread.join()

            # Get the result from the queue
            result = main_queue.get()

            if result[0]:
                self.utils.log.info('Program finished successfully')
                print('\n\nProgram finished successfully')
            else:
                self.utils.log.error('An error occurred: {}. Program aborted'.format(result[1]))
                print('\n\nAn error occurred: {}. Program aborted'.format(result[1]))

            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.utils.end_logs()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(session_id: int, interview_id: int):
    process = Process(session_id, interview_id)
    process.start_process()
    return {"status": "ok"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
