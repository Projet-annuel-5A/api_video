import torch
import torchaudio
import pandas as pd
from .utils import Utils
from typing import Dict
from pydub import AudioSegment


class AudioSplit:
    def __init__(self) -> None:
        self.utils = Utils()

    def __speech_to_text(self, filename: str, waveform: torch.Tensor, sampling_rate: int, lang: str) -> str:
        self.utils.log.info('Recognizing text from file {}'.format(filename))
        model_sampling_rate = self.utils.stt_processor.feature_extractor.sampling_rate
        resampler = torchaudio.transforms.Resample(sampling_rate, model_sampling_rate)
        resampled_waveform = resampler(waveform).squeeze().numpy()

        input_features = self.utils.stt_processor(resampled_waveform,
                                                  sampling_rate=model_sampling_rate,
                                                  return_tensors="pt").input_features

        predicted_ids = self.utils.stt_model.generate(input_features,
                                                      language=lang,
                                                      task="transcribe")

        transcription = self.utils.stt_processor.batch_decode(predicted_ids,
                                                              skip_special_tokens=True)

        return transcription[0].strip()

    '''
    def __lines_to_file(self, df: pd.DataFrame) -> None:
        path = os.path.join(self.utils.output_audio_folder, 'timing.txt')
        df.to_csv(path, sep='\t', index=False, header=False, encoding='utf-8')
        self.utils.log.info('Timeline for {} saved at {}'.format(self.utils.current_speaker, path))
    '''

    def __split_to_text(self, audiofile: AudioSegment, parts: Dict, lang: str) -> pd.DataFrame:
        self.utils.log.info('Start splitting audio file for {}'.format(self.utils.current_speaker))
        all_texts = pd.DataFrame(columns=['file', 'text'])

        for i in range(len(parts)):
            part_name = 'part_{:05d}'.format(i)

            start = parts[i][0] * 1000
            end = parts[i][1] * 1000
            split_audio = audiofile[start:end+500]

            self.utils.save_to_s3('part_{:05d}.wav'.format(i), split_audio.export(format='wav').read(),
                                  'audio', '{}/audioparts'.format(self.utils.current_speaker))

            tensor_audio = self.utils.audiosegment_to_tensor(split_audio)
            sampling_rate = split_audio.frame_rate
            text = self.__speech_to_text(part_name, tensor_audio, sampling_rate, lang)

            all_texts.loc[i] = [part_name, text]

        self.utils.log.info('End splitting {} for {}'.format(audiofile, self.utils.current_speaker))
        return all_texts

    def process(self, audiofile: AudioSegment, speakers: Dict, lang: str) -> pd.DataFrame:
        for speaker, lines in zip(speakers.keys(), speakers.values()):
            if speaker == self.utils.current_speaker:
                # TODO Save to database
                # self.__lines_to_file(lines)
                texts = self.__split_to_text(audiofile, lines, lang)
                texts.insert(0, 'speaker', self.utils.current_speaker)

        self.utils.log.info('Audio file splitted succesfully')
        return texts
