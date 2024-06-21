import uvicorn
import pandas as pd
from utils.models import Models
from audioEmotions import AudioEmotions
from fastapi import FastAPI, HTTPException

app = FastAPI()
models = Models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_audio")
async def process_audio(session_id: int, interview_id: int):
    ate = AudioEmotions(session_id=session_id,
                        interview_id=interview_id)
    try:
        res = pd.DataFrame(columns=['speaker', 'part', 'audio_emotions'])
        speakers = ate.utils.get_speakers_from_s3()
        for current_speaker in speakers:
            emotions = pd.DataFrame(columns=['speaker', 'part', 'audio_emotions'])
            all_files, all_emotions = ate.process_folder(current_speaker)
            emotions['part'] = all_files
            emotions['audio_emotions'] = all_emotions
            emotions['speaker'] = int(current_speaker.split('_')[1])
            res = pd.concat([res, emotions], ignore_index=True)

        ate.utils.df_to_temp_s3(res, filename='audio_emotions')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ate.utils.end_log()
        ate.utils.__del__()


@app.get("/testConfig")
def test_config():
    return {"Model '{}' loaded in".format(models.ate_model_id): models.device}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
