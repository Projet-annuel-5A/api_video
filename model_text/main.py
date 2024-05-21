import uvicorn
from textEmotions import TextEmotions
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_text")
async def process_text(session_id: int, interview_id: int):
    tte = TextEmotions(session_id=session_id,
                       interview_id=interview_id)
    try:
        # Open texts file from S3
        df = tte.utils.read_texts_from_s3()
        res = tte.process(df['text'])
        df['text_emotions'] = res

        tte.utils.df_to_temp_s3(df, filename='text_emotions')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tte.utils.end_log()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=True)
