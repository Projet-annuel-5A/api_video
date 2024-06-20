import uvicorn
from utils.models import Models
from videoEmotions import VideoEmotions
from fastapi import FastAPI, HTTPException

app = FastAPI()
models = Models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_video")
async def process_video(session_id: int, interview_id: int):
    vte = VideoEmotions(session_id=session_id,
                        interview_id=interview_id)
    try:
        speakers = vte.utils.get_speakers_from_s3()
        res = vte.process(speakers)
        vte.utils.df_to_temp_s3(res, filename='video_emotions')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        vte.utils.end_log()
        vte.utils.__del__()


@app.get("/testConfig")
def test_config():
    return {"Model loaded in": models.device}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=True)
