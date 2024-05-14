import torch
from .utils.utils import Utils


class TextEmotions:
    def __init__(self, session_id: str, interview_id: str, current_speaker: str) -> None:
        self.utils = Utils(session_id, interview_id, current_speaker)

    def process_texts(self, all_texts: list[str]) -> list[dict[str, float]]:
        self.utils.log.info('Start processing emotions from {} texts'.format(len(all_texts)))

        all_emotions = []
        for line in all_texts:
            inputs = self.utils.tte_tokenizer(line, return_tensors="pt")

            with torch.no_grad():
                logits = self.utils.tte_model(**inputs).logits

            # Get a list of the probabilities for each emotion
            values = torch.sigmoid(logits).squeeze(dim=0).tolist()

            # Get the percentage values and round them to 5 decimal places
            values = [round(num * 100, 5) for num in values]

            # Get a dictionary with the labels for each emotion and its values
            values_dict = dict(zip(self.utils.tte_model.config.id2label.values(), values))

            # Sort the dictionary by values in descending order
            sorted_values = {k: v for k, v in sorted(values_dict.items(), key=lambda x: x[1], reverse=True)}

            all_emotions.append(sorted_values)

        self.utils.log.info('All texts were processed')

        return all_emotions
