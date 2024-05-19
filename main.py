import os
import sys
import time
import tempfile
import threading
import pandas as pd
from tqdm import tqdm
from queue import Queue
from fastapi import FastAPI
from pydub import AudioSegment
from typing import Dict, Tuple
from middleware.utils.utils import Utils
from moviepy.editor import VideoFileClip
from model_text.textEmotions import TextEmotions
from middleware.utils.diarizator import Diarizator
from middleware.utils.audioSplit import AudioSplit
from model_audio.audioEmotions import AudioEmotions
from model_video.videoEmotions import VideoEmotions

app = FastAPI()


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


def __open_input_file(videoname: str, env: str) -> Tuple[AudioSegment, str, str] | None:
    if env == 'S3':
        s3_path = '{}/{}/raw/{}'.format(utils.session_id, utils.interview_id, videoname)
        try:
            video_bytes = utils.supabase_connection.download(s3_path)
            utils.log.info('Getting file {} from the S3 bucket'.format(videoname))
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file_path = temp_file.name
                try:
                    # Write the video_bytes to the temporary file
                    temp_file.write(video_bytes)
                    # Ensure data is written to disk
                    temp_file.flush()
                    utils.log.info('Starting audio extraction from video file')
                    # Open the video from the temporary file and extract the audio
                    audio = VideoFileClip(temp_file_path).audio
                    utils.log.info('Audio extraction finished')

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
            message = ('Error downloading the file {} from the S3 bucket: {}'.
                       format(videoname, e.args[0]['message']))
            utils.log.error(message)
            sys.exit(1)
        return audio_segment, temp_file_path, temp_file_path_2
    elif env == 'Local':
        filename = utils.config['GENERAL']['Filename']
        base_folder = os.path.join(os.path.dirname(__file__),
                                   utils.config['FOLDERS']['Main'],
                                   utils.session_id,
                                   utils.interview_id)
        video_path = os.path.join(base_folder, utils.config['FOLDERS']['Input'], filename)
        utils.log.info('File {} opened from local system'.format(videoname))
        video = VideoFileClip(video_path)
        utils.log.info('Starting audio extraction from video file')
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
        utils.log.info('Audio extraction finished')
        return audio_segment, filename, temp_file_path_2
    else:
        return None


def __analyze_text(all_texts: pd.DataFrame(), queue: Queue) -> None:
    all_texts['text_emotions'] = tte.process(all_texts['text'])

    # Save the results dataframe to a csv file
    # with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
    #    store.put('text', all_texts)

    queue.put(('emotions_from_text', all_texts))


def __analyse_video(_speakers: Dict, queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'video_emotions'])

    files, emotions = vte.process_folder(_speakers)
    results['file'] = files
    results['video_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    # with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
    #    store.put('video', results)

    queue.put(('emotions_from_video', results))


def __analyse_audio(queue: Queue) -> None:
    results = pd.DataFrame(columns=['speaker', 'file', 'audio_emotions'])

    files, emotions = ate.process_folder()

    results['file'] = files
    results['audio_emotions'] = emotions
    results['speaker'] = utils.current_speaker

    # with pd.HDFStore(os.path.join(utils.output_folder, 'results_{}.h5'.format(utils.current_speaker))) as store:
    #    store.put('audio', results)

    queue.put(('emotions_from_audio', results))


def __split_audio(_audiofile: AudioSegment, _speakers: Dict, lang: str = 'french') -> pd.DataFrame:
    utils.log.info('Starting split audio')
    texts = asp.process(_audiofile, _speakers, lang)
    utils.log.info('Split audio finished')
    return texts


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

    # Save the results file to S3
    filename = 'results.h5'
    s3_path = '{}/results/{}'.format(utils.output_s3_folder, filename)
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_file_path = temp_file.name
        try:
            # Write the file to the temporary file
            with pd.HDFStore(temp_file_path) as store:
                store.put('text', text_results)
                store.put('video', video_results)
                store.put('audio', audio_results)
                store.put('all', evaluations)

            with open(temp_file_path, 'rb') as f:
                try:
                    utils.supabase_connection.upload(file=f, path=s3_path,
                                                     file_options={'content-type': 'application/octet-stream'})
                    utils.log('File {} uploaded to S3 bucket at {}'.format(filename, s3_path))
                except Exception as e:
                    message = (
                        'Error uploading the file {} to the S3 bucket: {}'.format(filename, e.args[0]['message']))
                    utils.log(message)
        finally:
            temp_file.close()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    utils.log.info('Results merged successfully')


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
        audio_file, temp_file_path, temp_file_path_2 = __open_input_file(filename, 'S3')
        audio_name = '{}.wav'.format(filename.split('.')[0])
        # audio.write_audiofile('./files/{}'.format(audio_name))

        # Diarize and split the audio file
        speakers = drz.process(audio_file, audio_name)
        texts = __split_audio(audio_file, speakers, 'french')

        __analyze_text(texts, results_queue)
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

        __merge_results(evaluations, text_results, video_results, audio_results)
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
