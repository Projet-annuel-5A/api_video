import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from middleware.utils.utils import Utils


class VideoEmotions:
    def __init__(self) -> None:
        self.utils = Utils()

    def __predict(self, image: np.ndarray) -> dict[str, float]:
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

    def process_folder(self, _speakers: dict) -> tuple[list[str], list[dict[str, dict[str, float]]]]:
        all_files = list()
        all_emotions = list()

        parts = _speakers[self.utils.current_speaker]

        input_video_path = os.path.join(self.utils.input_folder, self.utils.config['GENERAL']['Filename'])
        clip = cv2.VideoCapture(input_video_path)

        # Get video fps
        fps = clip.get(cv2.CAP_PROP_FPS)

        # Set the interval for extracting frames
        timing = self.utils.config.getfloat('VIDEOEMOTION', 'Interval')
        interval = int(fps) * timing

        for i in range(len(parts)):
            video_emotions = {}
            part_name = 'part_{:05d}'.format(i)
            start_time = parts['start'][i]
            end_time = parts['end'][i]

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

        self.utils.log.info('Emotions extraction from video have finished')

        return all_files, all_emotions