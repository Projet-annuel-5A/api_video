import av
import io
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict
from deepface import DeepFace
from utils.utils import Utils
from dotenv import load_dotenv
from utils.models import Models


class VideoEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        """
        Initializes the VideoEmotions instance by loading environment variables and setting up
        utilities for video emotion analysis.
        Parameters:
            session_id (int): Session identifier to be used for processing.
            interview_id (int): Interview identifier for which video emotion analysis will be performed.
        """
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)

    def __predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predicts emotional content from a single frame using facial emotion recognition technology.
        Parameters:
            image (np.ndarray): The image array from a video frame.
        Returns:
            Dict[str, float]: A dictionary with emotion labels and their corresponding probabilities.
        Raises:
            ValueError: If no face is detected in the frame, returns a default response.
        """
        try:
            # ToDo: Replace library by the yolo model
            objs = DeepFace.analyze(image, actions=['emotion'])
            results = objs[0]['emotion']
            emotions = {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}
        except ValueError:
            emotions = {'No face detected': 0.0}
        return emotions

    def split_and_predict(self, segments: pd.DataFrame) -> List[List[Dict[str, float]]]:
        """
        Processes video segments to analyze emotions, extracting frames at specified intervals
        and applying emotion prediction.
        Parameters:
            segments (pd.DataFrame): DataFrame containing start and end times for video segments.
        Returns:
            List[List[Dict[str, float]]]: List of lists containing dictionaries with frame-specific emotion predictions.
        Raises:
            Exception: Captures and logs any exceptions, then re-raises them if video processing fails.
        """
        all_sentiments = list()

        print('Processing sentiments from video')
        self.utils.log.info('Recognizing emotions from video file')

        filename = self.utils.config['GENERAL']['Filename']
        s3_path = '{}/{}/raw/{}'.format(self.utils.session_id, self.utils.interview_id, filename)
        video_bytes = self.utils.open_input_file(s3_path, filename)
        container = av.open(io.BytesIO(video_bytes))

        # Set the interval for extracting frames
        timing = self.utils.config.getfloat('VIDEOEMOTION', 'Interval')

        try:
            for row in segments.itertuples():
                sentiments = list()
                image_count = 0

                start_time = row.start / 1000
                end_time = row.end / 1000

                container.seek(int(start_time * av.time_base))
                self.utils.log.info('Clip start time: {}, end time: {}.'.format(start_time, end_time))

                # Flag to ensure end_time frame is captured
                end_frame_captured = False

                # Initialize variables to track frame extraction
                last_extracted_time = start_time - timing  # ensures the first frame is extracted at start_time

                for frame in container.decode(video=0):
                    frame_time = frame.time

                    if frame_time < start_time:
                        continue

                    # Extract the last frame at end_time
                    if frame_time >= end_time and not end_frame_captured:
                        img = frame.to_image()
                        img_array = np.array(img)
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        sentiments.append({'frame_{:05d}'.format(image_count): self.__predict(img_array)})
                        image_count += 1

                        end_frame_captured = True
                        self.utils.log.info('Last frame captured: {}'.format(frame_time))
                        break

                    if start_time <= frame_time < end_time and frame_time >= last_extracted_time + timing:
                        img = frame.to_image()
                        img_array = np.array(img)
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        sentiments.append({'frame_{:05d}'.format(image_count): self.__predict(img_array)})
                        image_count += 1
                        self.utils.log.info('Capturing frame {}'.format(frame_time))

                        # Update the last extracted time
                        last_extracted_time = frame_time

                all_sentiments.append(sentiments)
        except Exception as e:
            message = ('Error processing video emotions.', str(e))
            self.utils.log.error(message)
            print(message)
        finally:
            container.close()
            self.utils.log.info('Emotions extraction from video have finished')
            print('Emotions extraction from video have finished')
            return all_sentiments
