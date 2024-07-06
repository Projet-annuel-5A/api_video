import os
import sys
import torch
import warnings
import configparser
warnings.filterwarnings("ignore", category=UserWarning)


class Models:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initializes the Models instance. If the instance is already initialized, it returns the existing instance
        (singleton pattern), otherwise, it sets up the configuration for the video processing module.
        Raises:
            IOError: If the configuration file 'videoConfig.ini' cannot be found, the program will terminate.
        """
        if not self.__initialized:
            self.config = self.__get_config()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.__initialized = True

    def __get_config(self) -> configparser.ConfigParser:
        """
        Reads and returns the configuration from 'videoConfig.ini', which contains
        necessary settings for video processing.
        Returns:
            configparser.ConfigParser: The loaded configuration settings.
        Raises:
            IOError: If the 'videoConfig.ini' file is not present, the method raises an IOError and exits the program.
        """
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'videoConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'videoConfig.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self):
        pass
