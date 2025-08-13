"""
Configuration settings for the Outfit Evaluator API
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
API_TITLE = "AI Outfit Evaluator API"
API_DESCRIPTION = "Analyze outfits and get AI-powered fashion suggestions"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# Model Paths
MODEL_PATH = BASE_DIR / "Models" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

# Model Settings
YOLO_MODEL_NAME = "best.pt"
CLIP_MODEL_NAME = "ViT-B/32"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# YOLO Class Names
CLASS_NAMES = [
    'sunglass', 'hat', 'jacket', 'shirt', 
    'pants', 'shorts', 'skirt', 'dress', 'bag', 'shoe'
]

# Occasion Mappings
OCCASIONS = {
    'job_interview': 'professional job interview',
    'date_night': 'romantic date night',
    'casual_hangout': 'casual social gathering',
    'work_meeting': 'business work meeting',
    'formal_event': 'formal wedding or gala event',
    'beach_vacation': 'beach or vacation setting',
    'night_out': 'night out with friends',
    'business_casual': 'business casual workplace'
}

# Scoring Weights
SCORING_WEIGHTS = {
    'clip_contextual': 0.6,
    'color_harmony': 0.2,
    'item_completeness': 0.1,
    'style_coherence': 0.1
}

# File Upload Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_MIME_TYPES = {
    'image/png', 'image/jpeg', 'image/jpg', 
    'image/gif', 'image/bmp'
}

# Color Detection Settings
MAX_COLORS_PER_ITEM = 3
COLOR_SIMILARITY_THRESHOLD = 40

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_PATH.parent, exist_ok=True)