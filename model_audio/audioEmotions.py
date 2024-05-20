import os
import torch
import tempfile
import torchaudio
import numpy as np
from utils.utils import Utils
import torch.nn.functional as f
from typing import Dict, List, Tuple


class AudioEmotions:
    def __init__(self, session_id: str, interview_id: str, current_speaker: str) -> None:
        self.utils = Utils(session_id, interview_id, current_speaker)

    def __speech_file_to_array_fn(self, path: str, sampling_rate: int) -> np.ndarray:
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()

        return speech

    def __predict(self, filename: str, path: str) -> Dict[str, float]:
        self.utils.log.info('Recognizing emotions from audio file {}'.format(filename))

        speech = self.__speech_file_to_array_fn(path, self.utils.ate_sampling_rate)
        inputs = self.utils.ate_feature_extractor(speech, sampling_rate=self.utils.ate_sampling_rate,
                                                  return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.utils.device) for key in inputs}

        with torch.no_grad():
            logits = self.utils.ate_model(**inputs).logits

        scores = f.softmax(logits, dim=1).detach().cpu().numpy()[0]

        # Get the percentage scores and round them to 5 decimal places
        scores = [round(num * 100, 5) for num in scores]

        # Get a dictionary with the labels for each emotion and its values
        values_dict = dict(zip(self.utils.ate_model.config.id2label.values(), scores))

        # Sort the dictionary by values in descending order
        sorted_values = {k: v for k, v in sorted(values_dict.items(), key=lambda x: x[1], reverse=True)}

        return sorted_values

    def process_folder(self) -> Tuple[List, List]:
        all_files = list()
        all_emotions = list()

        s3_path = '{}/{}/audioparts'.format(self.utils.output_s3_folder, self.utils.current_speaker)
        for file in self.utils.supabase_connection.list(s3_path):
            filename = file['name']
            if filename.split('.')[-1] == 'wav':
                downloaded_file = self.utils.supabase_connection.download('{}/{}'.format(s3_path, filename))
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file_path = temp_file.name

                    try:
                        temp_file.write(downloaded_file)
                        emotions = self.__predict(file, temp_file_path)
                    finally:
                        temp_file.close()
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

            all_emotions.append(emotions)
            all_files.append(filename.split('.')[0])

        return all_files, all_emotions
