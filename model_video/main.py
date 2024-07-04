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
        segments = vte.utils.get_segments_from_db()
        segments['video_emotions'] = vte.split_and_predict(segments)
        vte.utils.update_results(segments)
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
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)
