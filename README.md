# Video Emotions Analysis module

## Overview
The Video module of the **Interviewz** application processes video data to analyze emotional content using advanced computer vision models. It handles video segments, applies emotion recognition using frame analysis, and updates emotional data back to the database.

## Directory Structure
The module consists of several Python files organized as follows:
```plaintext
video/
├── app.py
├── videoEmotions.py
├── utils/
│   ├── models.py
│   ├── utils.py
```

## Components

### FastAPI Application (app.py)
Initializes a FastAPI application.

#### API Endpoints

```fastAPI
@app.get("/health")
"""
Returns the health status of the API. 
Description: Endpoint for checking the health status of the application.
Response: Returns a JSON object with the status "ok".
"""
```
```fastAPI
@app.post("/analyse_video")
"""
Endpoint to process video data for emotional analysis from stored video segments.
Parameters:
    session_id (int): The session ID for the video data.
    interview_id (int): The interview ID for the video data.
Returns:
    dict: A status message indicating the success or failure of the operation.
Raises:
    HTTPException: Exception with status code 500 indicating a server error if the process fails.
"""
```
```fastAPI
@app.post("/testConfig")
"""
Endpoint for testing the device where the models where loaded.
Response: JSON object showing the model ID and the device (CPU or GPU) it is loaded on.
"""
```

### VideoEmotions (videoEmotions.py):
Manages the video analysis process by fetching video data, applying frame-by-frame emotion analysis, and updating results.
Utilizes DeepFace for facial emotion recognition (planned to be replaced by a YOLO model).


### Utilities (utils/utils.py): 
Provides methods for configuration management, database interactions, file handling, and logging.
Manages connections to Supabase for data handling and S3 buckets for file storage.

### Models (utils/models.py):

Handles configuration settings but currently does not load specific models due to the module's reliance on DeepFace.