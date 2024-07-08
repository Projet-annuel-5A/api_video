import os
import sys
import torch
import warnings
import configparser
from ultralytics import YOLO
from google.cloud import storage, aiplatform
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
            self.backup_model = YOLO(self.__download_model())
            self.backup_model.to(self.device)
            self.predict_endpoint = self.__get_predict_endpoint()

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

    def __download_model(self) -> str:
        """
            Downloads a YOLO model from a cloud storage bucket and returns the loaded model instance.
            Returns:
                ultralytics.models.yolo.model.YOLO: The loaded YOLO model instance.
            Functionality:
                - Retrieves the model name from the configuration settings.
                - Establishes a connection to the cloud storage using the storage client.
                - Downloads the model file from the specified bucket to a local directory.
                - Loads, moves to the specified device and returns the YOLO model instance using the downloaded file.
            Raises:
                google.cloud.exceptions.GoogleCloudError: If there is an error in accessing the cloud storage.
                OSError: If there is an error in downloading or loading the model file.
            """
        try:
            model_name = self.config['BACKUPMODEL']['ModelName']
            bucket_name = self.config['GCLOUD']['BucketName']
            model_folder = self.config['BACKUPMODEL']['ModelFolder']
            model_path = os.path.join(model_folder, model_name)

            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(model_name)
            blob.download_to_filename(model_path)
            print('Model {} downloaded and loaded from bucket {}'.format(model_name, bucket_name))

            return model_path
        except Exception as e:
            print('Error downloading model : {}'.format(str(e)))

    def __get_predict_endpoint(self) -> aiplatform.Endpoint:
        """
        Retrieves the endpoint for the prediction service from the configuration settings.
        This method initializes the Google Cloud AI Platform with the specified project and location,
        then searches for an endpoint with the display name "yolo_predict".
        If found, it returns the endpoint object for making predictions.
        Returns:
            aiplatform.Endpoint: The endpoint object for the YOLO model predictions.
        Raises:
            ValueError: If the endpoint with the display name "yolo_predict" is not found.
        """
        aiplatform.init(project=self.config['GCLOUD']['ProjectName'],
                        location=self.config['GCLOUD']['Location'])
        endpoint_name = None
        for endpoint in aiplatform.Endpoint.list():
            if endpoint.display_name == "yolo_predict":
                endpoint_name = endpoint.name
        if endpoint_name is None:
            raise ValueError("Endpoint with display name 'yolo_predict' not found.")
        predict_endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        return predict_endpoint

    def check_endpoint(self) -> bool:
        """
        Checks if the traffic over the endpoint for the model is available.
        Returns:
            bool: True if the endpoint is available, False otherwise.
        """
        if not self.predict_endpoint.traffic_split:
            return False
        return True
