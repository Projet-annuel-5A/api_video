import os
import uvicorn
import pandas as pd
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from textEmotions import TextEmotions
from fastapi import FastAPI, HTTPException


app = FastAPI()


class InputItem(BaseModel):
    texts: List[str]


# Load environment variables from .env file
load_dotenv()

# Read startup parameters from environment variables
SESSION_ID = os.getenv("SESSION_ID")
INTERVIEW_ID = os.getenv("INTERVIEW_ID")
CURRENT_SPEAKER = 'speaker_00{}'.format(os.getenv("CURRENT_SPEAKER"))

# Initialize the TextEmotions class
tte = TextEmotions(session_id=SESSION_ID,
                   interview_id=INTERVIEW_ID,
                   current_speaker=CURRENT_SPEAKER)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_text")
async def process_text(texts: InputItem):
    try:
        # Convert input data to pandas Series
        all_texts = pd.Series(texts.texts)
        res = tte.process(all_texts)
        # Convert output series to list for JSON serialization
        return res.astype(str).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
