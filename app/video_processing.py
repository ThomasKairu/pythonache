import cv2
import numpy as np
from pytube import YouTube
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
from sqlalchemy.orm import Session
from . import crud
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    'field_color_lower': np.array([40, 40, 40]),
    'field_color_upper': np.array([70, 255, 255]),
    'motion_threshold': 30,
    'ball_detection_params': {
        'minDist': 20,
        'param1': 50,
        'param2': 30,
        'minRadius': 5,
        'maxRadius': 30
    },
    'goal_post_detection_params': {
        'threshold': 100,
        'minLineLength': 100,
        'maxLineGap': 10
    },
    'player_detection_scale': 1.05,
    'player_detection_win_stride': (8, 8),
    'player_detection_padding': (32, 32),
    'audio_excitement_threshold': 0.5,
    'highlight_threshold': 0.6,
    'weights': {
        'field': 0.2,
        'motion': 0.3,
        'ball': 0.2,
        'goal_post': 0.1,
        'players': 0.1,
        'audio': 0.2
    }
}

def download_youtube_video(url: str, output_path: str) -> str:
    """
    Download a YouTube video given its URL.
    
    :param url: URL of the YouTube video
    :param output_path: Directory to save the downloaded video
    :return: Path to the downloaded video file
    """
    try:
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        return video.download(output_path)
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return ""

def analyze_frame(frame: np.ndarray, prev_frame: np.ndarray, audio_segment: np.ndarray, fps: float, sample_rate: int) -> float:
    score = 0
    
    # 1. Field detection (green color)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CONFIG['field_color_lower'], CONFIG['field_color_upper'])
    green_percentage = np.sum(mask) / mask.size
    score += green_percentage * CONFIG['weights']['field']
    
    # 2. Motion detection
    if prev_frame is not None:
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, CONFIG['motion_threshold'], 255, cv2.THRESH_BINARY)
        motion_percentage = np.sum(motion_mask) / motion_mask.size
        score += motion_percentage * CONFIG['weights']['motion']
    
    # 3. Ball detection (circular objects)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, **CONFIG['ball_detection_params'])
    if circles is not None:
        score += CONFIG['weights']['ball']
    
    # 4. Goal post detection (vertical lines)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, **CONFIG['goal_post_detection_params'])
    if lines is not None:
        vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]  # Nearly vertical lines
        if len(vertical_lines) > 0:
            score += CONFIG['weights']['goal_post']
    
    # 5. Player detection (using HOG descriptor)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    _, _ = hog.detectMultiScale(frame, 
                                scale=CONFIG['player_detection_scale'], 
                                winStride=CONFIG['player_detection_win_stride'], 
                                padding=CONFIG['player_detection_padding'])
    if len(_) > 0:
        score += CONFIG['weights']['players'] * min(len(_) / 10, 1)  # Cap at 1
    
    # 6. Audio analysis (if audio is available)
    if audio_segment is not None and len(audio_segment) > 0:
        # Compute audio features
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)
        
        # Check for sudden increases in volume or pitch (potential excitement)
        if np.mean(mfccs[1:]) > CONFIG['audio_excitement_threshold'] or np.mean(spectral_centroid) > 1000:
            score += CONFIG['weights']['audio']
    
    return min(score, 1.0)  # Ensure the score is between 0 and 1

def process_frame(args):
    frame_num, frame, prev_frame, audio_segment, fps, sample_rate = args
    score = analyze_frame(frame, prev_frame, audio_segment, fps, sample_rate)
    return frame_num, score

class HighlightClassifier:
    def __init__(self, model_path='highlight_classifier.joblib'):
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_features(self, frame: np.ndarray, audio_segment: np.ndarray, fps: float, sample_rate: int) -> np.ndarray:
        features = []
        
        # Extract visual features
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        features.append(np.mean(hsv[:,:,1]))  # Mean saturation
        features.append(np.mean(hsv[:,:,2]))  # Mean value (brightness)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(np.std(gray))  # Standard deviation of pixel intensities
        
        # Extract motion features (assuming we have access to the previous frame)
        if hasattr(self, 'prev_frame'):
            frame_diff = cv2.absdiff(frame, self.prev_frame)
            features.append(np.mean(frame_diff))
        else:
            features.append(0)
        self.prev_frame = frame
        
        # Extract audio features
        if len(audio_segment) > 0:
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)
            features.append(np.mean(spectral_centroid))
        else:
            features.extend([0] * 14)  # Placeholder for audio features if not available
        
        return np.array(features)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy}")
        joblib.dump(self.model, self.model_path)

    def predict(self, features):
        return self.model.predict([features])[0]

def detect_highlights(video_path: str, db: Session, job_id: int) -> List[Tuple[float, float]]:
    highlights = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load audio
    audio, sample_rate = librosa.load(video_path)
    
    classifier = HighlightClassifier()
    
    with ThreadPoolExecutor() as executor:
        frame_scores = []
        futures = []
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_num / fps
            
            # Get corresponding audio segment
            audio_start = int(current_time * sample_rate)
            audio_end = int((current_time + 1/fps) * sample_rate)
            audio_segment = audio[audio_start:audio_end]
            
            future = executor.submit(process_frame_ml, (frame_num, frame, audio_segment, fps, sample_rate, classifier))
            futures.append(future)
            
            # Process results in batches to avoid memory issues
            if len(futures) >= 100 or frame_num == total_frames - 1:
                for future in futures:
                    frame_num, is_highlight = future.result()
                    frame_scores.append((frame_num, is_highlight))
                futures = []
            
            # Update progress every 5%
            if frame_num % (total_frames // 20) == 0:
                progress = 25 + int(50 * frame_num / total_frames)
                crud.update_job(db, job_id, "processing", progress)
    
    cap.release()
    
    # Process scores to detect highlights
    frame_scores.sort(key=lambda x: x[0])  # Sort by frame number
    start_time = None
    for frame_num, is_highlight in frame_scores:
        current_time = frame_num / fps
        if is_highlight:
            if start_time is None:
                start_time = current_time
        elif start_time is not None:
            highlights.append((start_time, current_time))
            start_time = None
    
    # Add final highlight if video ends during a highlight
    if start_time is not None:
        highlights.append((start_time, total_frames / fps))
    
    return highlights

def process_frame_ml(args):
    frame_num, frame, audio_segment, fps, sample_rate, classifier = args
    features = classifier.extract_features(frame, audio_segment, fps, sample_rate)
    is_highlight = classifier.predict(features)
    return frame_num, is_highlight

def train_highlight_classifier(video_paths: List[str], highlight_annotations: List[List[Tuple[float, float]]]):
    classifier = HighlightClassifier()
    X = []
    y = []
    
    for video_path, annotations in zip(video_paths, highlight_annotations):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        audio, sample_rate = librosa.load(video_path)
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_num / fps
            
            audio_start = int(current_time * sample_rate)
            audio_end = int((current_time + 1/fps) * sample_rate)
            audio_segment = audio[audio_start:audio_end]
            
            features = classifier.extract_features(frame, audio_segment, fps, sample_rate)
            X.append(features)
            
            is_highlight = any(start <= current_time <= end for start, end in annotations)
            y.append(1 if is_highlight else 0)
        
        cap.release()
    
    classifier.train(np.array(X), np.array(y))
    logger.info("Highlight classifier trained successfully.")

def create_highlight_video(input_path: str, output_path: str, highlights: list):
    """
    Create a highlight video from the original video.
    
    :param input_path: Path to the original video file
    :param output_path: Path to save the highlight video
    :param highlights: List of tuples containing start and end times of highlights
    """
    video = VideoFileClip(input_path)
    highlight_clips = [video.subclip(start, end) for start, end in highlights]
    final_clip = concatenate_videoclips(highlight_clips)
    final_clip.write_videofile(output_path)

def process_video(db: Session, url: str, output_dir: str, job_id: int):
    """
    Main function to process a video: download, detect highlights, and create highlight video.
    
    :param db: Database session
    :param url: URL of the YouTube video
    :param output_dir: Directory to save the output files
    :param job_id: ID of the job in the database
    """
    try:
        # Update job status to "downloading"
        crud.update_job(db, job_id, "downloading", 0)

        # Download the video
        video_path = download_youtube_video(url, output_dir)
        if not video_path:
            crud.update_job(db, job_id, "failed", 0)
            return "Failed to download video"

        # Update job status to "processing"
        crud.update_job(db, job_id, "processing", 25)

        # Detect highlights
        highlights = detect_highlights(video_path, db, job_id)

        # Update job status to "creating highlight video"
        crud.update_job(db, job_id, "creating highlight video", 75)

        # Create highlight video
        highlight_video_path = os.path.join(output_dir, "highlight_video.mp4")
        create_highlight_video(video_path, highlight_video_path, highlights)

        # Update job status to "completed"
        crud.update_job(db, job_id, "completed", 100)

        return f"Highlight video created at {highlight_video_path}"
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        crud.update_job(db, job_id, "failed", 0)
        return f"Error processing video: {str(e)}"