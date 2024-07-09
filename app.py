import uvicorn
from utils.models import Models
from videoEmotions import VideoEmotions
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application lifespan...")
    models = Models()
    print('Starting on : {}'.format(models.device))

    yield
    print('Closing server...')


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """
    Returns the health status of the API.
    Description: Endpoint for checking the health status of the application.
    Response: Returns a JSON object with the status "ok".
    """
    return {"status": "ok"}


@app.post("/analyse_video")
async def process_video(session_id: int, interview_id: int, model_type: str):
    """
    Endpoint to process video data for emotional analysis from stored video segments.
    Parameters:
        session_id (int): The session ID for the video data.
        interview_id (int): The interview ID for the video data.
        model_type (str): The model type to use for prediction. Should be either 'cloud' or 'local'.
    Returns:
        dict: A status message indicating the success or failure of the operation.
    Raises:
        HTTPException: Exception with status code 500 indicating a server error if the process fails.
    """
    vte = VideoEmotions(session_id=session_id,
                        interview_id=interview_id,
                        model_type=model_type)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)
