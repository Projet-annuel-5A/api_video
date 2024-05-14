import os
import pandas as pd
from .utils import Utils
from typing import List, Dict
from pydub import AudioSegment


class AudioSplit:
    def __init__(self) -> None:
        self.utils = Utils()

    def __lines_to_file(self, df: pd.DataFrame) -> None:
        path = os.path.join(self.utils.output_audio_folder, 'timing.txt')
        df.to_csv(path, sep='\t', index=False, header=False, encoding='utf-8')
        self.utils.log.info('Timeline for {} saved at {}'.format(self.utils.current_speaker, path))

    def __multiple_split(self, audiofile: str, parts: pd.DataFrame) -> None:
        self.utils.log.info('Reading audiosegments from audio file {}'.format(audiofile))
        audio = AudioSegment.from_wav(os.path.join(self.utils.temp_folder, audiofile))
        self.utils.log.info('Start splitting {} for {}'.format(audiofile, self.utils.current_speaker))
        for i in range(len(parts)):
            start = parts.loc[i, "start"] * 1000
            end = parts.loc[i, "end"] * 1000
            split_audio = audio[start:end+500]
            split_audio.export(os.path.join(self.utils.output_audio_folder,
                                            'part_{:05d}.wav'.format(i)), format="wav")
        self.utils.log.info('End splitting {} for {}'.format(audiofile, self.utils.current_speaker))

    def process(self, filename: str, speakers: Dict) -> None:
        self.utils.log.info('Start splitting {}'.format(filename))

        for speaker, df in zip(speakers.keys(), speakers.values()):
            if speaker == self.utils.current_speaker:
                self.__lines_to_file(df)
                self.__multiple_split(filename, df)

        self.utils.log.info('File {} splitted succesfully'.format(filename))
