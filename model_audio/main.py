import json

import uvicorn
from audioEmotions import AudioEmotions
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process_folder")
async def process_folder(session_id, interview_id, current_speaker):
    ate = AudioEmotions(session_id=session_id, interview_id=interview_id, current_speaker=current_speaker)
    all_files, all_emotions = ate.process_folder()
    #return json.dumps([all_files, all_emotions])
    json_compatible_item_data = jsonable_encoder([all_files, all_emotions])
    return JSONResponse(content=json_compatible_item_data)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
