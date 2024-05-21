import time
import json
import uvicorn
import requests
import threading
import pandas as pd
from tqdm import tqdm
from typing import Dict
from queue import Queue
from .utils.utils import Utils
from dotenv import load_dotenv
from pydub import AudioSegment
from .utils.diarizator import Diarizator
from .utils.audioSplit import AudioSplit
from fastapi import FastAPI, HTTPException

global utils

app = FastAPI()

# Load environment variables from .env file
load_dotenv()


def __analyze_text(params: Dict, queue: Queue) -> None:
    url = 'http://127.0.0.1:8002/analyse_text'
    response = requests.post(url, params=params)

    if response.status_code == 200:
        # TODO update value in DB
        queue.put(('emotions_from_text', True))
    else:
        utils.log.error("Error:", response.status_code)
        queue.put(('emotions_from_text', False))


def __analyse_video(params: Dict, queue: Queue) -> None:
    url = 'http://127.0.0.1:8003/analyse_video'
    response = requests.post(url, params=params)

    if response.status_code == 200:
        # TODO update value in DB
        queue.put(('emotions_from_video', True))
    else:
        utils.log.error("Error:", response.status_code)
        queue.put(('emotions_from_video', False))


def __analyse_audio(params: Dict, queue: Queue) -> None:
    url = 'http://127.0.0.1:8001/analyse_audio'
    response = requests.post(url, params=params)

    if response.status_code == 200:
        # TODO update value in DB
        queue.put(('emotions_from_audio', True))
    else:
        utils.log.error("Error:", response.status_code)
        queue.put(('emotions_from_audio', False))


def __split_audio(_audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> None:
    asp = AudioSplit()
    utils.log.info('Starting split audio')
    texts = asp.process(_audiofile, _speakers, lang)
    # Save texts to S3
    utils.df_to_temp_s3(texts, filename='texts')
    utils.log.info('Split audio finished')
    # TODO update diarization champ in DB


def __process_all(session_id: int, interview_id: int, queue: Queue) -> None:
    drz = Diarizator()
    text_results = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions'])
    video_results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])
    audio_results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])
    params = {
        'session_id': session_id,
        'interview_id': interview_id
    }

    try:
        # Create a queue to store the results
        results_queue = Queue()
        filename = utils.config['GENERAL']['Filename']
        # Extract the audio from the video file
        audio_file, temp_file_path, temp_file_path_2 = utils.open_input_file(filename)
        audio_name = '{}.wav'.format(filename.split('.')[0])

        # Diarize and split the audio file
        speakers = drz.process(audio_file, audio_name)
        utils.save_to_s3('speakers.json', json.dumps(speakers).encode(), 'text', 'temp')
        __split_audio(audio_file, speakers, utils.config['GENERAL']['Language'])

        # Define the processing threads
        thread_process_text = threading.Thread(target=__analyze_text, args=(params, results_queue))
        thread_process_audio = threading.Thread(target=__analyse_audio, args=(params, results_queue))
        thread_process_video = threading.Thread(target=__analyse_video, args=(params, results_queue))

        # Start the threads
        utils.log.info('Starting emotions detection threads from text, audio and video')
        thread_process_text.start()
        thread_process_audio.start()
        thread_process_video.start()

        # Wait for all threads to finish
        thread_process_text.join()
        thread_process_audio.join()
        thread_process_video.join()
        utils.log.info('Emotions detection threads from text, audio and video have finished')
        while not results_queue.empty():
            thread_id, result = results_queue.get()
            if thread_id == 'emotions_from_text':
                text_results = result
            elif thread_id == 'emotions_from_video':
                video_results = result
            elif thread_id == 'emotions_from_audio':
                audio_results = result

        if text_results and video_results and audio_results:
            utils.merge_results()

        result = (True, None)
    except Exception as e:
        result = (False, e)

    finally:
        utils.delete_temp_files([temp_file_path, temp_file_path_2])
    queue.put(result)


def process(session_id: int, interview_id: int):
    try:
        print('Program started')

        utils = Utils(session_id, interview_id)
        main_queue = Queue()

        utils.log.info("Program started => Session: {} | Interview: {}".format(session_id, interview_id))

        # Start the process function in a separate thread
        main_thread = threading.Thread(target=__process_all, args=(session_id, interview_id, main_queue, ))
        main_thread.start()

        # Create an indeterminate progress bar
        with tqdm(desc='Work in progress', unit='iterations', leave=True) as progress_bar:
            i = 0
            while main_thread.is_alive():
                progress_bar.write('\rProcessing... {}'.format('*' * i), end='', nolock=True)
                i += 1
                time.sleep(2)
                progress_bar.refresh()

        # Wait for the main thread to finish
        main_thread.join()

        # Get the result from the queue
        result = main_queue.get()

        if result[0]:
            utils.log.info('Program finished successfully')
            print('\n\nProgram finished successfully')
        else:
            utils.log.error('An error occurred: {}. Program aborted'.format(result[1]))
            print('\n\nAn error occurred: {}. Program aborted'.format(result[1]))

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        utils.end_logs()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(session_id: int, interview_id: int):
    process(session_id, interview_id)
    return {"status": "ok"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
