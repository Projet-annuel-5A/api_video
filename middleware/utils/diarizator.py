import os
import torch
import torchaudio
import pandas as pd
from .utils import Utils
from typing import Dict, Tuple
from pyannote.core.annotation import Annotation


class Diarizator:
    def __init__(self) -> None:
        self.utils = Utils()

    def __read_audio(self, filename: str) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(filename)
        self.utils.log.info('File {} opened'.format(filename))
        return waveform, sample_rate

    def __split_diarization(self, diarization: Annotation, num_speakers: int) -> Dict[str, pd.DataFrame]:
        self.utils.log.info('Start splitting diarization')

        speakers_dict = {}
        start = 0
        end = 0
        current_speaker = ''

        for i in range(num_speakers):
            speakers_dict['speaker_' + str('{:03d}'.format(i))] = pd.DataFrame(columns=['start', 'end'])

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if current_speaker == '':
                start = turn.start
                end = turn.end
                current_speaker = speaker
            else:
                if speaker == current_speaker:
                    end = turn.end
                else:
                    key = 'speaker_{:03d}'.format(int(current_speaker.split('_')[1]))
                    df = speakers_dict[key]
                    df.loc[len(df)] = [start, end]
                    speakers_dict[key] = df

                    start = turn.start
                    end = turn.end
                    current_speaker = speaker

        key = 'speaker_{:03d}'.format(int(current_speaker.split('_')[1]))
        df = speakers_dict[key]
        df.loc[len(df)] = [start, end]
        speakers_dict[key] = df

        self.utils.log.info('Split completed successfully')

        return speakers_dict

    def __diarization_to_file(self, diarization: Annotation, audiofile: str) -> None:
        filename = os.path.join(self.utils.output_folder, '{}.rttm'.format(audiofile.split('.')[0]))
        with open(filename, "w") as rttm:
            diarization.write_rttm(rttm)

    def __diarize(self, audiofile: str) -> Annotation:
        audiofile = os.path.join(self.utils.temp_folder, audiofile)
        pipeline = self.utils.diarization_pipeline
        waveform, sample_rate = self.__read_audio(audiofile)
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        self.utils.log.info('Diarization completed successfully')
        return diarization

    def process(self, filename: str) -> Dict[str, pd.DataFrame]:
        self.utils.log.info('Start diarization over {}'.format(filename))
        diarization = self.__diarize(filename)
        self.utils.log.info(diarization)
        self.__diarization_to_file(diarization, filename)
        speakers = self.__split_diarization(diarization, 2)
        self.utils.log.info('File {} diarizated succesfully'.format(filename))
        return speakers
