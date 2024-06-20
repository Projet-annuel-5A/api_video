import os
import torch
import tempfile
import torchaudio
from utils.utils import Utils
from dotenv import load_dotenv
from utils.models import Models
import torch.nn.functional as f
from typing import Dict, List, Tuple


class AudioEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)
        self.models = Models()

    def process_folder(self, current_speaker: str) -> Tuple[List, List]:
        all_files = list()
        all_emotions = list()

        s3_path = '{}/{}'.format(self.utils.output_s3_folder, current_speaker)
        for file in self.utils.supabase_connection.list(s3_path):
            filename = file['name']
            if filename.split('.')[-1] == 'wav':

                emotions = self.__download_and_predict(filename, s3_path)
                all_emotions.append(emotions)
                part_number = int((filename.split('.')[0]).split('_')[-1])
                all_files.append(part_number)
        return all_files, all_emotions

    def __download_and_predict(self, filename: str, s3_path: str) -> Dict[str, float]:
        self.utils.log.info('Recognizing emotions from audio file {}'.format(filename))

        downloaded_file = self.utils.supabase_connection.download('{}/{}'.format(s3_path, filename))
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(downloaded_file)

        speech_array, _sampling_rate = torchaudio.load(temp_file_path)

        resampler = torchaudio.transforms.Resample(_sampling_rate, self.models.ate_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()

        inputs = self.models.ate_feature_extractor(speech, sampling_rate=self.models.ate_sampling_rate,
                                                   return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.models.device) for key in inputs}

        with torch.no_grad():
            logits = self.models.ate_model(**inputs).logits

        scores = f.softmax(logits, dim=1).detach().cpu().numpy()[0]

        # Get the percentage scores and round them to 5 decimal places
        scores = [round(num * 100, 5) for num in scores]

        # Get a dictionary with the labels for each emotion and its values
        values_dict = dict(zip(self.models.ate_model.config.id2label.values(), scores))

        # Sort the dictionary by values in descending order
        sorted_values = {k: v for k, v in sorted(values_dict.items(), key=lambda x: x[1], reverse=True)}

        temp_file.close()
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return sorted_values
