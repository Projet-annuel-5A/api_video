import torch
import pandas as pd
from typing import Dict
from utils.utils import Utils
from dotenv import load_dotenv
from utils.models import Models


class TextEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)
        self.models = Models()

    def __process_text(self, text: str) -> Dict[str, float]:
        inputs = self.models.tte_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.models.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.models.tte_model(**inputs).logits

        # Get a list of the probabilities for each emotion
        values = torch.sigmoid(logits).squeeze(dim=0).tolist()

        # Get the percentage values and round them to 5 decimal places
        values = [round(num * 100, 5) for num in values]

        # Get a dictionary with the labels for each emotion and its values
        values_dict = dict(zip(self.models.tte_model.config.id2label.values(), values))

        # Sort the dictionary by values in descending order
        sorted_values = {k: v for k, v in sorted(values_dict.items(), key=lambda x: x[1], reverse=True)}

        return sorted_values

    def process(self, all_texts: pd.Series) -> pd.Series:
        self.utils.log.info('Start processing emotions from {} texts'.format(all_texts.size))
        text_emotions = all_texts.apply(self.__process_text)
        return text_emotions
