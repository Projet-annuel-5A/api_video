import uvicorn
from utils.models import Models
from textEmotions import TextEmotions
from fastapi import FastAPI, HTTPException

app = FastAPI()
models = Models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_text")
async def process_text(session_id: int, interview_id: int):
    tte = TextEmotions(session_id=session_id,
                       interview_id=interview_id)
    try:
        # Open texts file from database
        res = tte.utils.get_texts_from_db()
        res['text_emotions'] = tte.process(res['text'])
        res.drop(columns='text', inplace=True)
        tte.utils.update_results(res)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tte.utils.end_log()
        tte.utils.__del__()


@app.get("/testConfig")
def test_config():
    return {"Model '{}' loaded in".format(models.tte_model_id): models.device}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
