import pandas as pd
from .utils import Utils
from moviepy.editor import *


class VideoProcess:

    def __init__(self) -> None:
        self.utils = Utils()

    def to_audio_old(self, video_path: str, audio_path: str) -> None:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.utils.log.info('Starting audio extraction from video file')

        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)

        self.utils.log.info('Audio extraction finished. Audio file saved at {}'.format(audio_path))

        video.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def __multiple_split_old(self, videofile: str, parts: pd.DataFrame) -> None:
        self.utils.log.info('Start splitting {} for {}'.format(videofile, self.utils.current_speaker))

        video = VideoFileClip(os.path.join(self.utils.input_folder, videofile)).set_audio(None)

        for i in range(len(parts)):
            start_time = parts['start'][i]
            end_time = parts['end'][i]
            path = os.path.join(self.utils.output_video_folder, 'part_{:05d}.mp4'.format(i))
            clip = video.subclip(start_time, end_time)
            clip.write_videofile(path)
            self.utils.log.info('Part {:05d} saved at {}'.format(
                i, os.path.join(path)
            ))
            clip.close()

        video.close()
        '''sys.stdout = original_stdout
        sys.stderr = original_stderr'''

    def split_video_old(self, speakers: dict) -> None:
        filename = self.utils.config['GENERAL']['Filename']
        for speaker, df in zip(speakers.keys(), speakers.values()):
            if speaker == self.utils.current_speaker:
                self.__multiple_split(filename, df)
