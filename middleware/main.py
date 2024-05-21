import os
import time
import json
import uvicorn
import requests
import threading
import pandas as pd
from tqdm import tqdm
from queue import Queue
from .utils.utils import Utils
from dotenv import load_dotenv
from pydub import AudioSegment
from typing import Dict, Tuple, List
from .utils.diarizator import Diarizator
from .utils.audioSplit import AudioSplit
from fastapi import FastAPI, HTTPException
# from model_text.textEmotions import TextEmotions
# from model_audio.audioEmotions import AudioEmotions
# from model_video.videoEmotions import VideoEmotions


app = FastAPI()

# Load environment variables from .env file
load_dotenv()


def __init_all(session_id: int, interview_id: int) -> None:
    global utils
    global asp
    global drz
    # global ate
    # global tte
    # global vte

    utils = Utils(session_id, interview_id)
    asp = AudioSplit()
    # ate = AudioEmotions(session_id, interview_id, speaker_name)
    drz = Diarizator()
    # tte = TextEmotions(session_id, interview_id, speaker_name)
    # vte = VideoEmotions(session_id, interview_id, speaker_name)


def __analyze_text(all_texts: pd.DataFrame(), queue: Queue) -> None:
    # all_texts['text_emotions'] = tte.process(all_texts['text'])

    url = 'http://127.0.0.1:8002/analyse_text'
    payload = {
        "texts": all_texts['text'].tolist()
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        all_texts['text_emotions'] = response.json()
    else:
        utils.log.error("Error:", response.status_code)

    queue.put(('emotions_from_text', all_texts))


def __analyse_video(_speakers: Dict[str, List[Tuple[float, float]]], queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])

    # files, emotions = vte.process_folder(_speakers)

    url = 'http://127.0.0.1:8003/analyse_video'
    payload = {
        "speakers": json.dumps(_speakers)
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        results['file'] = response.json()['all_files']
        results['video_emotions'] = json.loads(response.json()['all_emotions'])
        results['speaker'] = utils.current_speaker
    else:
        utils.log.error("Error:", response.status_code)

    # results['file'] = files
    # results['video_emotions'] = emotions

    queue.put(('emotions_from_video', results))


def __analyse_audio(queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    # files, emotions = ate.process_folder()

    url = 'http://127.0.0.1:8001/analyse_audio'
    response = requests.get(url)

    if response.status_code == 200:
        results['file'] = response.json()['file']
        results['audio_emotions'] = response.json()['audio_emotions']
        results['speaker'] = utils.current_speaker
    else:
        utils.log.error("Error:", response.status_code)

    # results['file'] = files
    # results['audio_emotions'] = emotions
    # results['speaker'] = utils.current_speaker

    queue.put(('emotions_from_audio', results))


def __split_audio(_audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> None:
    utils.log.info('Starting split audio')
    texts = asp.process(_audiofile, _speakers, lang)
    # Save texts to S3
    utils.df_to_temp_s3(texts, filename='texts')
    utils.log.info('Split audio finished')
    # TODO update diarization champ in DB


def __process_all(queue: Queue) -> None:
    evaluations = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions', 'video_emotions', 'audio_emotions'])
    text_results = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions'])
    video_results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])
    audio_results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    try:
        # Create a queue to store the results
        results_queue = Queue()
        filename = utils.config['GENERAL']['Filename']
        # Extract the audio from the video file
        # TODO Change env
        audio_file, temp_file_path, temp_file_path_2 = utils.open_input_file(filename, 'S3')
        audio_name = '{}.wav'.format(filename.split('.')[0])

        # Diarize and split the audio file
        speakers = drz.process(audio_file, audio_name)
        utils.save_to_s3('speakers.json', json.dumps(speakers).encode(), 'text', 'temp')
        __split_audio(audio_file, speakers, utils.config['GENERAL']['Language'])
        '''
        # Define the processing threads
        thread_process_text = threading.Thread(target=__analyze_text, args=(texts, results_queue,))
        thread_process_audio = threading.Thread(target=__analyse_audio, args=(results_queue, ))
        thread_process_video = threading.Thread(target=__analyse_video, args=(speakers, results_queue))

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

        utils.merge_results(evaluations, text_results, video_results, audio_results)
    '''
        result = (True, None)
    except Exception as e:
        result = (False, e)

    # TODO uncomment block
    '''
    finally:
        utils.delete_temp_files([temp_file_path, temp_file_path_2])
    '''
    queue.put(result)


def process(session_id: int, interview_id: int):
    try:
        print('Program started')

        __init_all(session_id, interview_id)
        main_queue = Queue()

        utils.log.info("Program started => Session: {} | Interview: {}".format(session_id, interview_id))

        # Start the process function in a separate thread
        main_thread = threading.Thread(target=__process_all, args=(main_queue, ))
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
