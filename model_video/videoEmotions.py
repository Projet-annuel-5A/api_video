import os
import cv2
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict
from deepface import DeepFace
from utils.utils import Utils
from dotenv import load_dotenv
from utils.models import Models


class VideoEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)

    def __predict(self, image: np.ndarray) -> Dict[str, float]:
        try:
            # ToDo: Replace library by the yolo model
            objs = DeepFace.analyze(image, actions=['emotion'])
            results = objs[0]['emotion']
            emotions = {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}
        except ValueError:
            emotions = {'No face detected': 0.0}
        return emotions

    def split_and_predict(self, segments: pd.DataFrame) -> List[List[Dict[str, float]]]:
        all_sentiments = list()

        print('Processing sentiments from video')
        self.utils.log.info('Recognizing emotions from video file')

        filename = self.utils.config['GENERAL']['Filename']
        s3_path = '{}/{}/raw/{}'.format(self.utils.session_id, self.utils.interview_id, filename)
        video_bytes = self.utils.open_input_file(s3_path, filename)

        with (tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file):
            temp_file_path = temp_file.name
            try:
                temp_file.write(video_bytes)
                clip = cv2.VideoCapture(temp_file_path)

                # Get video fps
                fps = clip.get(cv2.CAP_PROP_FPS)

                # Set the interval for extracting frames
                timing = self.utils.config.getfloat('VIDEOEMOTION', 'Interval')
                interval = int(fps) * timing

                for row in segments.itertuples():
                    sentiments = list()

                    # Calculate frame indices for starting and ending times
                    start_frame = int(row.start / 1000 * fps)
                    end_frame = int(row.end / 1000 * fps)

                    # Set starting frame
                    clip.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    # Read the video frame by frame and send respective frames to prediction
                    frame_count = 0
                    image_count = 0
                    while clip.isOpened() and frame_count <= (end_frame - start_frame):
                        ret, frame = clip.read()

                        # If there are no more frames, break the loop
                        if not ret:
                            break

                        # Detect emotions from the frame if it's the first one or if it's a multiple of the interval
                        if frame_count == 0 or frame_count % interval == 0:
                            image_name = 'image_{:05d}'.format(image_count)
                            image_count += 1

                            sentiments.append({image_name: self.__predict(frame)})

                        # Save the last frame
                        elif start_frame + frame_count == end_frame:
                            image_name = 'image_{:05d}'.format(image_count)
                            image_count += 1

                            sentiments.append({image_name: self.__predict(frame)})

                        frame_count += 1

                    all_sentiments.append(sentiments)

                # Release the video capture object
                clip.release()
            except Exception as e:
                message = ('Error processing video emotions.', str(e))
                self.utils.log.error(message)
                print(message)
            finally:
                temp_file.close()
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            self.utils.log.info('Emotions extraction from video have finished')
            print('Emotions extraction from video have finished')
            return all_sentiments
