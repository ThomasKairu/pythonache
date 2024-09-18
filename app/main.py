from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from . import crud, models, storage, schemas, video_processing
from .database import engine, get_db
from typing import List

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Football Highlights Application API"}

@app.post("/upload/", response_model=schemas.Video)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = storage.save_upload_file(file)
    if file_path:
        video = crud.create_video(db, title=file.filename, file_path=file_path, duration=0)  # Duration to be updated later
        return video
    raise HTTPException(status_code=400, detail="Failed to upload video")

@app.get("/videos/", response_model=List[schemas.Video])
def read_videos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    videos = crud.get_videos(db, skip=skip, limit=limit)
    return videos

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-video/", response_model=schemas.Job)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    url: str,
    output_dir: str,
    db: Session = Depends(get_db)
):
    # Create a new job
    job = crud.create_job(db, video_id=None)  # You might want to associate this with a video later

    # Add the processing task to background tasks
    background_tasks.add_task(video_processing.process_video, db, url, output_dir, job.id)

    return job

@app.get("/jobs/{job_id}", response_model=schemas.Job)
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/videos/{video_id}", response_model=schemas.Video)
def read_video(video_id: int, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id=video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@app.put("/videos/{video_id}/status", response_model=schemas.Video)
def update_video_status(video_id: int, status: schemas.VideoStatusUpdate, db: Session = Depends(get_db)):
    updated_video = crud.update_video_status(db, video_id=video_id, new_status=status.status)
    if updated_video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return updated_video

@app.put("/videos/{video_id}", response_model=schemas.Video)
def update_video(video_id: int, video: schemas.VideoUpdate, db: Session = Depends(get_db)):
    updated_video = crud.update_video(db, video_id=video_id, video_data=video.dict(exclude_unset=True))
    if updated_video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return updated_video

@app.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    result = crud.delete_video(db, video_id=video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return result