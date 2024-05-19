import torch
import pandas as pd
from .utils import Utils
from typing import Dict
from pydub import AudioSegment
from pyannote.core.annotation import Annotation


class Diarizator:
    def __init__(self) -> None:
        self.utils = Utils()

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

    '''
    def __diarization_to_file(self, diarization: Annotation, audiofile: str) -> None:
        filename = os.path.join(self.utils.output_folder, '{}.rttm'.format(audiofile.split('.')[0]))
        with open(filename, "w") as rttm:
            diarization.write_rttm(rttm)
    '''

    def __diarize(self, waveform: torch.Tensor, sample_rate: int) -> Annotation:
        pipeline = self.utils.diarization_pipeline
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        self.utils.log.info('Diarization completed successfully')
        return diarization

    def process(self, audio_file: AudioSegment, filename: str) -> Dict[str, pd.DataFrame]:
        self.utils.log.info('Start diarization over audio file')
        # Convert the audio in Torch Tensor
        waveform = self.utils.audiosegment_to_tensor(audio_file)
        sampling_rate = audio_file.frame_rate
        diarization = self.__diarize(waveform, sampling_rate)
        self.utils.log.info(diarization)
        diarization_str = diarization.to_rttm().encode()
        self.utils.save_to_s3('{}.rttm'.format(filename.split('.')[0]), diarization_str, 'text')
        speakers = self.__split_diarization(diarization, 2)
        self.utils.log.info('File {} diarizated succesfully'.format(filename))
        return speakers
