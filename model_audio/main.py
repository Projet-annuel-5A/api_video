import os
import uvicorn
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from audioEmotions import AudioEmotions
from fastapi import FastAPI, HTTPException


app = FastAPI()


class OutputModel(BaseModel):
    file: List[str]
    audio_emotions: List[Dict[str, float]]


# Load environment variables from .env file
load_dotenv()

# Read startup parameters from environment variables
SESSION_ID = os.getenv("SESSION_ID")
INTERVIEW_ID = os.getenv("INTERVIEW_ID")
CURRENT_SPEAKER = 'speaker_00{}'.format(os.getenv("CURRENT_SPEAKER"))

# Initialize the TextEmotions class
ate = AudioEmotions(session_id=SESSION_ID,
                    interview_id=INTERVIEW_ID,
                    current_speaker=CURRENT_SPEAKER)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyse_audio", response_model=OutputModel)
async def process_audio():
    try:
        all_files, all_emotions = ate.process_folder()
        res = OutputModel(
            file=all_files,
            audio_emotions=all_emotions
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/end_audio_log")
async def end_audio_log():
    try:
        ate.utils.end_log()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
