import uvicorn
from videoEmotions import VideoEmotions
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse_video")
async def process_video(session_id: int, interview_id: int):
    vte = VideoEmotions(session_id=session_id,
                        interview_id=interview_id)
    print('Vte instance created')
    try:
        speakers = vte.utils.get_speakers_from_s3()
        print('Speakers:', speakers)
        res = vte.process(speakers)
        print('Finished processing')
        vte.utils.df_to_temp_s3(res, filename='video_emotions')
        print('Dataframe saved')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        vte.utils.end_log()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=True)
