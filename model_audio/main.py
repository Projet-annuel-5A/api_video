import uvicorn
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
        segments = ate.utils.get_segments_from_db()
        segments['audio_emotions'] = ate.split_and_predict(segments)
        ate.utils.update_results(segments)
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
