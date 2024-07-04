import io
import torch
import torchaudio
import pandas as pd
from typing import Dict, List
from utils.utils import Utils
from dotenv import load_dotenv
from pydub import AudioSegment
from utils.models import Models
import torch.nn.functional as f


class AudioEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)
        self.models = Models()

    def split_and_predict(self, segments: pd.DataFrame) -> List[Dict[str, float]]:
        sentiments = list()

        try:
            filename = self.utils.config['GENERAL']['Audioname']
            self.utils.log.info('Recognizing emotions from audio file')
            s3_path = '{}/{}/raw/{}'.format(self.utils.session_id, self.utils.interview_id, filename)
            audio_bytes = self.utils.open_input_file(s3_path, filename)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

            for row in segments.itertuples():
                audio_segment = audio[row.start:row.end]
                audio_segment_bytes = io.BytesIO()
                audio_segment.export(audio_segment_bytes, format="mp3")
                audio_segment_bytes.seek(0)

                speech_array, sample_rate = torchaudio.load(audio_segment_bytes)
                resampler = torchaudio.transforms.Resample(sample_rate, self.models.ate_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()

                inputs = self.models.ate_feature_extractor(speech,
                                                           sampling_rate=self.models.ate_sampling_rate,
                                                           return_tensors="pt",
                                                           padding=True)

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

                sentiments.append(sorted_values)
        except Exception as e:
            message = ('Error splitting and predicting the emotions from the audio file.', str(e))
            self.utils.log.error(message)
            raise e

        return sentiments
