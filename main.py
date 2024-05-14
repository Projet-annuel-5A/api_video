import time
import threading
import pandas as pd
from tqdm import tqdm
from queue import Queue
from fastapi import FastAPI
from moviepy.editor import *
from middleware.utils.utils import Utils
from middleware.utils.diarizator import Diarizator
from middleware.utils.audioSplit import AudioSplit
from middleware.utils.videoProcess import VideoProcess
from middleware.utils.speechToText import SpeechToText
from model_text.textEmotions import TextEmotions
from model_audio.audioEmotions import AudioEmotions
from model_video.videoEmotions import VideoEmotions

app = FastAPI()

# TODO Move main from root to middleware folder


def __video_to_audio(video_path: str, audio_path: str) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    utils.log.info('Starting audio extraction from video file')

    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    utils.log.info('Audio extraction finished. Audio file saved at {}'.format(audio_path))

    video.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr


def __analyze_text(queue: Queue) -> None:
    stt = SpeechToText()
    tte = TextEmotions()

    results = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions'])

    files, texts = stt.process_folder('french')

    text_emotions = tte.process_texts(texts)

    results['file'] = files
    results['text'] = texts
    results['text_emotions'] = text_emotions
    results['speaker'] = utils.current_speaker

    # Save the results dataframe to a csv file
    with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
        store.put('text', results)

    queue.put(('emotions_from_text', results))


def __analyse_video(_speakers: dict, queue: Queue) -> None:
    vte = VideoEmotions()

    results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])

    files, emotions = vte.process_folder(_speakers)
    results['file'] = files
    results['video_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
        store.put('video', results)

    queue.put(('emotions_from_video', results))


def __analyse_audio(queue: Queue) -> None:
    ate = AudioEmotions()

    results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    files, emotions = ate.process_folder()

    results['file'] = files
    results['audio_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
        store.put('audio', results)

    queue.put(('emotions_from_audio', results))


def __split_audio(_audiofile: str, _speakers: dict) -> None:
    utils.log.info('Starting split audio')
    asp = AudioSplit()

    asp.process(_audiofile, _speakers)
    utils.log.info('Split audio finished')


def __split_video_parts_old(_speakers: dict) -> None:
    vpp = VideoProcess()

    vpp.split_video_old(_speakers)


def __merge_results(evaluations: pd.DataFrame,
                    text_results: pd.DataFrame,
                    video_results: pd.DataFrame,
                    audio_results: pd.DataFrame) -> None:
    utils.log.info('Merging results from text, audio and video processing for speaker {}'.format(utils.current_speaker))
    evaluations = pd.concat([evaluations, text_results], ignore_index=True)

    for index, row in video_results.iterrows():
        # Find the corresponding row in evaluations based on matching values of columns speaker and file
        mask = (evaluations['speaker'] == row['speaker']) & (evaluations['file'] == row['file'])
        # Update values of column video_emotions in evaluations with corresponding values from video_results
        evaluations.loc[mask, 'video_emotions'] = (evaluations.loc[mask, 'video_emotions'].
                                                   apply(lambda x: row['video_emotions']))

    for index, row in audio_results.iterrows():
        # Find the corresponding row in evaluations based on matching values of columns speaker and file
        mask = (evaluations['speaker'] == row['speaker']) & (evaluations['file'] == row['file'])
        # Update values of column audio_emotions in evaluations with corresponding values from audio_results
        evaluations.loc[mask, 'audio_emotions'] = (evaluations.loc[mask, 'audio_emotions'].
                                                   apply(lambda x: row['audio_emotions']))

    results_path = os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))
    with pd.HDFStore(results_path) as store:
        store.put('all', evaluations)

    utils.log.info('Results merged successfully, saved to {}'.format(results_path))


def __process_all(queue: Queue) -> None:
    drz = Diarizator()

    evaluations = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions', 'video_emotions', 'audio_emotions'])
    text_results = pd.DataFrame(columns=['speaker', 'file', 'text', 'text_emotions'])
    video_results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])
    audio_results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    try:
        # Create a queue to store the results
        results_queue = Queue()

        # Split the audio and video files
        filename = utils.config['GENERAL']['Filename']
        video_path = os.path.join(utils.input_folder, filename)
        audio_name = '{}.wav'.format(filename.split('.')[0])
        audio_path = os.path.join(utils.temp_folder, audio_name)
        __video_to_audio(video_path, audio_path)

        # Diarize and split the audio file
        speakers = drz.process(audio_name)
        __split_audio(audio_name, speakers)

        # Define the processing threads
        thread_process_text = threading.Thread(target=__analyze_text, args=(results_queue,))
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

        __merge_results(evaluations, text_results, video_results, audio_results)

        result = (True, None)

    except Exception as e:
        result = (False, e)

    queue.put(result)


# @app.post("/process")
def process(session_id: str, interview_id: str, current_speaker: str) -> None:
    print('Program started')

    speaker_name = 'speaker_00{}'.format(current_speaker)
    global utils
    utils = Utils(session_id, interview_id, speaker_name)
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
    utils.delete_temp_files()


# TODO Convert to api
if __name__ == '__main__':
    _session_id = sys.argv[1] if len(sys.argv) > 1 else None
    _interview_id = sys.argv[2] if len(sys.argv) > 2 else None
    _current_speaker = sys.argv[3] if len(sys.argv) > 3 else None
    process(_session_id, _interview_id, _current_speaker)
    # test(_session_id, _interview_id)
    # uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
