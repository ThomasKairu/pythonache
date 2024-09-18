from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Video schemas

class VideoBase(BaseModel):
    title: str
    file_path: str
    duration: int
    status: str

class VideoCreate(VideoBase):
    pass

class VideoUpdate(BaseModel):
    title: Optional[str] = None
    file_path: Optional[str] = None
    duration: Optional[int] = None
    status: Optional[str] = None

class VideoStatusUpdate(BaseModel):
    status: str

class Video(VideoBase):
    id: int

    class Config:
        orm_mode = True

# Job schemas

class JobBase(BaseModel):
    video_id: Optional[int] = None
    status: str
    progress: int

class JobCreate(JobBase):
    pass

class JobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None

class Job(JobBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Response models

class VideoResponse(Video):
    jobs: list[Job] = []

class JobResponse(Job):
    video: Optional[Video] = None