import time
import json
import uvicorn
import requests
import threading
from tqdm import tqdm
from typing import Dict
from queue import Queue
import concurrent.futures
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
        self.increasing_tqdm = False
        self.session_id = session_id
        self.interview_id = interview_id
        self.utils = Utils(session_id, interview_id)
        self.params = {
            'session_id': self.session_id,
            'interview_id': self.interview_id
        }

    def __analyse_text(self) -> Dict[str, bool]:
        url = 'http://127.0.0.1:8002/analyse_text'
        response = requests.post(url, params=self.params)
        if response.status_code == 200:
            self.utils.update_bool_db('text_ok', True)
            print('\nText analysis done')
            self.increasing_tqdm = False
            return {'text': True}
        else:
            self.utils.log.error("Error:", response.status_code)
            return {'text': False}

    def __analyse_video(self) -> Dict[str, bool]:
        url = 'http://127.0.0.1:8003/analyse_video'
        response = requests.post(url, params=self.params)

        if response.status_code == 200:
            self.utils.update_bool_db('video_ok', True)
            print('\nVideo analysis done')
            self.increasing_tqdm = False
            return {'video': True}
        else:
            self.utils.log.error("Error:", response.status_code)
            return {'video': False}

    def __analyse_audio(self) -> Dict[str, bool]:
        url = 'http://127.0.0.1:8001/analyse_audio'
        response = requests.post(url, params=self.params)

        if response.status_code == 200:
            self.utils.update_bool_db('audio_ok', True)
            print('\nAudio analysis done')
            self.increasing_tqdm = False
            return {'audio': True}
        else:
            self.utils.log.error("Error:", response.status_code)
            return {'audio': False}

    def __split_audio(self, _audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> None:
        asp = AudioSplit()
        self.utils.log.info('Starting split audio')
        print('Starting split audio')
        texts = asp.process(_audiofile, _speakers, lang)

        # Save texts to S3
        print('\nAudio split successfully, saving texts to S3')
        self.utils.df_to_temp_s3(texts, filename='texts')
        self.utils.update_bool_db('audio_split_ok', True)
        self.utils.log.info('Split audio finished')

    def __process_all(self, queue: Queue) -> None:
        drz = Diarizator()
        temp_files = []

        all_results = []
        try:

            filename = self.utils.config['GENERAL']['Filename']
            # Extract the audio from the video file
            audio_file, temp_file_path, temp_file_path_2 = self.utils.open_input_file(filename)
            temp_files.append(temp_file_path)
            temp_files.append(temp_file_path_2)
            audio_name = '{}.wav'.format(filename.split('.')[0])

            # Diarize and split the audio file
            speakers = drz.process(audio_file, audio_name)
            print('\nDiarization done')
            self.increasing_tqdm = False
            self.utils.save_to_s3('speakers.json', json.dumps(speakers).encode(), 'text', 'temp')
            self.__split_audio(audio_file, speakers, self.utils.config['GENERAL']['Language'])
            print('\nSplit audio done')
            self.increasing_tqdm = False

            self.utils.log.info('Starting emotions detection from text, audio and video')
            print('\nStarting emotions detection from text, audio and video')

            # Submit the three methods to be executed concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                text_results = executor.submit(self.__analyse_text)
                audio_results = executor.submit(self.__analyse_audio)
                video_results = executor.submit(self.__analyse_video)

            for future in concurrent.futures.as_completed([text_results, audio_results, video_results]):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    self.utils.log.error('An error occurred: {}'.format(e))
                    print('\nAn error occurred: {}'.format(e))

            self.utils.log.info('Emotions detection threads from text, audio and video have finished')
            result = (True, None)
        except Exception as e:
            result = (False, e)
        finally:
            print('\nMerging and saving results')
            self.increasing_tqdm = False
            self.utils.merge_results(all_results)

            print('\nDeleting temporary files')
            self.increasing_tqdm = False
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
                    if not self.increasing_tqdm:
                        i = 0
                        progress_bar.reset()
                        self.increasing_tqdm = True
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
            print('\nSaving log files')
            self.utils.end_logs()
            print('Program finished')


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(session_id: int, interview_id: int):
    process = Process(session_id, interview_id)
    process.start_process()
    return {"status": "ok"}


@app.post("/delete_all")
async def delete_all(session_id: int, interview_id: int):
    load_dotenv()
    utils = Utils(session_id, interview_id)
    try:
        # Delete all files from S3
        # List subfolders and files in the root folder
        path = '{}/{}/output'.format(session_id, interview_id)
        content = ([file['name'] for file in utils.supabase_connection.list(path)])
        subfolders = [folder for folder in content if '.' not in folder]
        files_to_delete = ['{}/{}'.format(path, file) for file in content if '.' in file]

        # List files to delete in subfolders
        for folder in subfolders:
            files_to_delete += (['{}/{}/{}'.format(path, folder, file['name'])
                                 for file in utils.supabase_connection.list('{}/{}'.format(path, folder))])

        for file in files_to_delete:
            utils.supabase_connection.remove(file)

        # Delete results in DB
        utils.supabase.table('results').delete().eq('interview_id', interview_id).execute()

        # Update fields to false
        utils.update_bool_db('text_ok', False)
        utils.update_bool_db('video_ok', False)
        utils.update_bool_db('audio_ok', False)
        utils.update_bool_db('diarization_ok', False)

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
