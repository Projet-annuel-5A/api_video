import os
import torchaudio
from .utils import Utils
from typing import List, Tuple


class SpeechToText:
    def __init__(self) -> None:
        self.utils = Utils()

    def __process_file(self, filename: str, lang: str = 'french') -> str:
        self.utils.log.info('Recognizing text from file {}'.format(filename))
        model_sampling_rate = self.utils.stt_processor.feature_extractor.sampling_rate
        waveform, sampling_rate = torchaudio.load(filename)
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

    def process_folder(self, lang: str = 'french') -> Tuple[List[str], List[str]]:
        all_files = []
        all_texts = []

        for file in os.listdir(self.utils.output_audio_folder):
            if file.split('.')[-1] == 'wav':
                file_to_process = os.path.join(self.utils.output_audio_folder, file)
                text = self.__process_file(file_to_process, lang)
                all_texts.append(text)
                all_files.append(file.split('.')[0])

        with open(os.path.join(self.utils.output_folder,
                               '{}_texts.txt'.format(self.utils.current_speaker)), 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_texts))

        return all_files, all_texts
