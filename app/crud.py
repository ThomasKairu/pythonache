from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from . import models
from fastapi import HTTPException
import datetime

# Video CRUD operations

def create_video(db: Session, title: str, file_path: str, duration: int):
    try:
        db_video = models.Video(title=title, file_path=file_path, duration=duration, status="uploaded")
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
        return db_video
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_video(db: Session, video_id: int):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

def get_videos(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Video).offset(skip).limit(limit).all()

def update_video_status(db: Session, video_id: int, new_status: str):
    try:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if db_video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        db_video.status = new_status
        db.commit()
        db.refresh(db_video)
        return db_video
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def update_video(db: Session, video_id: int, video_data: dict):
    try:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if db_video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        for key, value in video_data.items():
            setattr(db_video, key, value)
        db.commit()
        db.refresh(db_video)
        return db_video
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def delete_video(db: Session, video_id: int):
    try:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if db_video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        db.delete(db_video)
        db.commit()
        return {"message": "Video deleted successfully"}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Job CRUD operations

def create_job(db: Session, video_id: int = None):
    try:
        db_job = models.Job(video_id=video_id, status="pending", progress=0)
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_job(db: Session, job_id: int):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

def update_job(db: Session, job_id: int, status: str, progress: int):
    try:
        db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if db_job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        db_job.status = status
        db_job.progress = progress
        db_job.updated_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(db_job)
        return db_job
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def get_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Job).offset(skip).limit(limit).all()

def delete_job(db: Session, job_id: int):
    try:
        db_job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if db_job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        db.delete(db_job)
        db.commit()
        return {"message": "Job deleted successfully"}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")