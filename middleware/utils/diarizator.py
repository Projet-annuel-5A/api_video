import torch
from .utils import Utils
from pydub import AudioSegment
from typing import Dict, List, Tuple
from pyannote.core.annotation import Annotation


class Diarizator:
    def __init__(self) -> None:
        self.utils = Utils()

    def __split_diarization(self, diarization: Annotation) -> Dict[str, List[Tuple[float, float]]]:
        self.utils.log.info('Start splitting diarization')
        speakers_dict = {}
        '''
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker = 'speaker_{:03d}'.format(int(speaker.split('_')[1]))
            if current_speaker == '':
                current_speaker = speaker
                values = (turn.start, turn.end)
            elif speaker == current_speaker:
                values = (values[0], turn.end)
            else:
                if speakers_dict.get(current_speaker) is None:
                    speakers_dict[current_speaker] = list()
                speakers_dict[current_speaker].append(values)
                current_speaker = speaker
                values = (turn.start, turn.end)
        if speakers_dict.get(current_speaker) is None:
            speakers_dict[current_speaker] = list()
        speakers_dict[current_speaker].append(values)
        '''
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            current_speaker = 'speaker_{:03d}'.format(int(speaker.split('_')[1]))
            values = (turn.start, turn.end)
            if speakers_dict.get(current_speaker) is None:
                speakers_dict[current_speaker] = list()
            speakers_dict[current_speaker].append(values)

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

    def process(self, audio_file: AudioSegment, filename: str) -> Dict[str, List[Tuple[float, float]]]:
        self.utils.log.info('Start diarization over audio file')
        # Convert the audio in Torch Tensor
        waveform = self.utils.audiosegment_to_tensor(audio_file)
        sampling_rate = audio_file.frame_rate
        diarization = self.__diarize(waveform, sampling_rate)
        self.utils.log.info(diarization)
        diarization_str = diarization.to_rttm().encode()
        self.utils.save_to_s3('{}.rttm'.format(filename.split('.')[0]), diarization_str, 'text')
        speakers = self.__split_diarization(diarization)
        self.utils.log.info('File {} diarizated succesfully'.format(filename))
        return speakers
