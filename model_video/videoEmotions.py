import os
import cv2
import torch
import tempfile
import numpy as np
import torch.nn as nn
from utils.utils import Utils
from typing import List, Dict, Tuple


class VideoEmotions:
    def __init__(self, session_id: str, interview_id: str, current_speaker: str) -> None:
        self.utils = Utils(session_id, interview_id, current_speaker)

    def __predict(self, image: np.ndarray) -> Dict[str, float]:
        inputs = self.utils.vte_processor(image, return_tensors="pt")
        vte_model = self.utils.vte_model(**inputs)
        logits = vte_model.logits
        attention_weights = vte_model.attentions

        m = nn.Softmax(dim=0)
        values = m(logits[0])

        w_values = np.zeros(len(values))
        for i in range(len(w_values)):
            result = values[i].item() * attention_weights[i]
            w_values[i] = torch.sum(result)

        # Convert to percentage
        total = np.sum(w_values)
        w_values = w_values * 100 / total

        output = dict(zip(self.utils.vte_model.config.id2label.values(), w_values))

        sorted_values = {k: v for k, v in sorted(output.items(), key=lambda x: x[1], reverse=True)}

        return sorted_values

    def process_folder(self, _speakers: Dict) -> Tuple[List[str], List[Dict[str, Dict[str, float]]]]:
        all_files = list()
        all_emotions = list()

        parts = _speakers[self.utils.current_speaker]

        s3_path = '{}/{}/raw/{}'.format(self.utils.session_id,
                                        self.utils.interview_id,
                                        self.utils.config['GENERAL']['Filename'])
        video_bytes = self.utils.supabase_connection.download(s3_path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                temp_file.write(video_bytes)

                clip = cv2.VideoCapture(temp_file_path)
                # Get video fps
                fps = clip.get(cv2.CAP_PROP_FPS)

                # Set the interval for extracting frames
                timing = self.utils.config.getfloat('VIDEOEMOTION', 'Interval')
                interval = int(fps) * timing

                for i in range(len(parts)):
                    video_emotions = {}
                    part_name = 'part_{:05d}'.format(i)
                    start_time = parts[i][0]
                    end_time = parts[i][1]

                    # Calculate frame indices for starting and ending times
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)

                    # Set starting frame
                    clip.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    # Read the video frame by frame and send respective frames to prediction
                    frame_count = 0
                    image_count = 0
                    while clip.isOpened() and frame_count <= (end_frame - start_frame):
                        image_name = 'image_{:05d}'.format(image_count)
                        ret, frame = clip.read()

                        # If there are no more frames, break the loop
                        if not ret:
                            break

                        # Detect emotions from the frame if it's a multiple of the interval
                        if frame_count % interval == 0:
                            video_emotions[image_name] = self.__predict(frame)
                            image_count += 1
                            last_frame = None
                        else:
                            last_frame = frame
                        frame_count += 1

                        # Save the last frame
                        if last_frame is not None:
                            video_emotions[image_name] = self.__predict(last_frame)

                    # Release the video capture object
                    all_emotions.append(video_emotions)
                    all_files.append(part_name)
                clip.release()
            finally:
                temp_file.close()
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        self.utils.log.info('Emotions extraction from video have finished')

        return all_files, all_emotions
