import os
import json
import uvicorn
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from videoEmotions import VideoEmotions
from fastapi import FastAPI, HTTPException


app = FastAPI()


class OutputModel(BaseModel):
    all_files: List[str]
    all_emotions: str


# Load environment variables from .env file
load_dotenv()

# Read startup parameters from environment variables
SESSION_ID = os.getenv("SESSION_ID")
INTERVIEW_ID = os.getenv("INTERVIEW_ID")
CURRENT_SPEAKER = 'speaker_00{}'.format(os.getenv("CURRENT_SPEAKER"))

# Initialize the TextEmotions class
vte = VideoEmotions(session_id=SESSION_ID,
                    interview_id=INTERVIEW_ID,
                    current_speaker=CURRENT_SPEAKER)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_video", response_model=OutputModel)
async def process_video(speakers: str):
    try:
        all_speakers = json.loads(speakers)
        all_files, all_emotions = vte.process_folder(all_speakers)
        return {"all_files": all_files, "all_emotions": json.dumps(all_emotions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8009, reload=True)
