import torch
import torchaudio
import pandas as pd
from typing import Dict
from .utils import Utils
from .models import Models
from pydub import AudioSegment


class AudioSplit:
    def __init__(self) -> None:
        self.utils = Utils()
        self.models = Models()

    def __speech_to_text(self, part: int, waveform: torch.Tensor, sampling_rate: int, lang: str) -> str:
        self.utils.log.info('Recognizing text from part {}'.format(part))

        model_sampling_rate = self.models.stt_processor.feature_extractor.sampling_rate

        resampler = torchaudio.transforms.Resample(sampling_rate, model_sampling_rate)
        resampler = resampler.to(self.models.device)

        # Resample the waveform
        resampled_waveform = resampler(waveform).squeeze()
        # Move the resampled waveform to the CPU before converting to numpy
        resampled_waveform = resampled_waveform.cpu().numpy()

        input_features = self.models.stt_processor(resampled_waveform,
                                                   sampling_rate=model_sampling_rate,
                                                   return_tensors="pt").input_features
        input_features = input_features.to(self.models.device)

        predicted_ids = self.models.stt_model.generate(input_features,
                                                       language=lang,
                                                       task="transcribe")

        transcription = self.models.stt_processor.batch_decode(predicted_ids,
                                                               skip_special_tokens=True)

        return transcription[0].strip()

    def __split_to_text(self, audiofile: AudioSegment, parts: Dict, current_speaker: str, lang: str) -> pd.DataFrame:
        self.utils.log.info('Start splitting audio for {}'.format(current_speaker))
        all_texts = pd.DataFrame(columns=['part', 'start', 'end', 'text'])

        for i in range(len(parts)):
            start = parts[i][0] * 1000
            end = parts[i][1] * 1000
            split_audio = audiofile[start:end+500]

            self.utils.save_to_s3('part_{:05d}.wav'.format(i), split_audio.export(format='wav').read(),
                                  'audio', current_speaker)

            tensor_audio = self.utils.audiosegment_to_tensor(split_audio)
            tensor_audio = tensor_audio.to(self.models.device)
            sampling_rate = split_audio.frame_rate
            text = self.__speech_to_text(i, tensor_audio, sampling_rate, lang)

            all_texts.loc[i] = [i, start, end, text]

        self.utils.log.info('End splitting {} for {}'.format(audiofile, current_speaker))

        return all_texts

    def process(self, audiofile: AudioSegment, speakers: Dict, lang: str) -> pd.DataFrame:
        all_texts = pd.DataFrame(columns=['speaker', 'part', 'start', 'end', 'text'])
        for speaker, lines in zip(speakers.keys(), speakers.values()):
            self.utils.log.info('Processing speaker {}'.format(speaker))
            texts = self.__split_to_text(audiofile, lines, speaker, lang)
            texts.insert(0, 'speaker', int(speaker.split('_')[1]))
            all_texts = pd.concat([all_texts, texts], ignore_index=True)

        self.utils.log.info('Audio file split successfully')
        return all_texts
