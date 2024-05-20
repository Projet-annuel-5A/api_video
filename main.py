import os
import sys
import time
import threading
import pandas as pd
from tqdm import tqdm
from queue import Queue
from fastapi import FastAPI
from pydub import AudioSegment
from dotenv import load_dotenv
from typing import Dict, Tuple, List
from middleware.utils.utils import Utils
from model_text.textEmotions import TextEmotions
from middleware.utils.diarizator import Diarizator
from middleware.utils.audioSplit import AudioSplit
from model_audio.audioEmotions import AudioEmotions
from model_video.videoEmotions import VideoEmotions


app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Read startup parameters from environment variables
SESSION_ID = os.getenv("SESSION_ID")
INTERVIEW_ID = os.getenv("INTERVIEW_ID")
CURRENT_SPEAKER = 'speaker_00{}'.format(os.getenv("CURRENT_SPEAKER"))


def __init_all(session_id: str, interview_id: str, speaker_name: str) -> None:
    global utils
    global asp
    global ate
    global drz
    global tte
    global vte

    utils = Utils(session_id, interview_id, speaker_name)
    asp = AudioSplit()
    ate = AudioEmotions(session_id, interview_id, speaker_name)
    drz = Diarizator()
    tte = TextEmotions(session_id, interview_id, speaker_name)
    vte = VideoEmotions(session_id, interview_id, speaker_name)


def __analyze_text(all_texts: pd.DataFrame(), queue: Queue) -> None:
    all_texts['text_emotions'] = tte.process(all_texts['text'])

    queue.put(('emotions_from_text', all_texts))


def __analyse_video(_speakers: Dict[str, List[Tuple[float, float]]], queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])

    # TODO Sent json.dumps(_speakers) to the API
    files, emotions = vte.process_folder(_speakers)
    results['file'] = files
    # TODO results['video_emotions'] = json.loads(emotions)
    results['video_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    queue.put(('emotions_from_video', results))


def __analyse_audio(queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    files, emotions = ate.process_folder()

    results['file'] = files
    results['audio_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    queue.put(('emotions_from_audio', results))


def __split_audio(_audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> pd.DataFrame:
    utils.log.info('Starting split audio')
    texts = asp.process(_audiofile, _speakers, lang)
    utils.log.info('Split audio finished')
    return texts


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
        texts = __split_audio(audio_file, speakers, 'french')

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
        result = (True, None)

    except Exception as e:
        result = (False, e)

    utils.delete_temp_files([temp_file_path, temp_file_path_2])
    queue.put(result)


# @app.post("/process")
def process(session_id: str, interview_id: str, current_speaker: str) -> None:
    print('Program started')

    speaker_name = 'speaker_00{}'.format(current_speaker)
    __init_all(session_id, interview_id, speaker_name)
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

    utils.end_logs()
    ate.utils.end_logs()
    tte.utils.end_logs()
    vte.utils.end_logs()


# TODO Convert to api
if __name__ == '__main__':
    _session_id = sys.argv[1] if len(sys.argv) > 1 else None
    _interview_id = sys.argv[2] if len(sys.argv) > 2 else None
    _current_speaker = sys.argv[3] if len(sys.argv) > 3 else None
    process(_session_id, _interview_id, _current_speaker)
    # test(_session_id, _interview_id)
    # uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
