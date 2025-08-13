"""
Model loading and initialization service
"""

import torch
import clip
import google.generativeai as genai
from ultralytics import YOLO
from typing import Optional, Tuple, Any
import os

from app.config import (
    MODEL_PATH, GEMINI_API_KEY, CLIP_MODEL_NAME, 
    GEMINI_MODEL_NAME, CLASS_NAMES
)

class ModelLoader:
    """Handles loading and initialization of all AI models"""
    
    def __init__(self):
        """Initialize model loader"""
        self.yolo_model: Optional[YOLO] = None
        self.clip_model: Optional[Any] = None
        self.clip_preprocess: Optional[Any] = None
        self.gemini_model: Optional[Any] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Model loader initialized. Device: {self.device}")
    
    def load_yolo_model(self) -> bool:
        """
        Load YOLO detection model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading YOLO model from: {MODEL_PATH}")
            
            if not os.path.exists(MODEL_PATH):
                print(f"Error: Model file not found at {MODEL_PATH}")
                return False
            
            self.yolo_model = YOLO(str(MODEL_PATH))
            print("YOLO model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    
    def load_clip_model(self) -> bool:
        """
        Load CLIP model for contextual analysis
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
            
            self.clip_model, self.clip_preprocess = clip.load(
                CLIP_MODEL_NAME, 
                device=self.device
            )
            
            print(f"CLIP model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            return False
    
    def load_gemini_model(self) -> bool:
        """
        Configure and load Gemini model for LLM suggestions
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
                print("Warning: Gemini API key not configured")
                return False
            
            print("Configuring Gemini model...")
            
            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Initialize model
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            print("Gemini model configured successfully")
            return True
            
        except Exception as e:
            print(f"Error configuring Gemini model: {e}")
            return False
    
    def load_all_models(self) -> dict:
        """
        Load all models and return status
        
        Returns:
            dict: Status of each model loading attempt
        """
        print("Loading all models...")
        
        status = {
            'yolo': self.load_yolo_model(),
            'clip': self.load_clip_model(),
            'gemini': self.load_gemini_model()
        }
        
        # Summary
        successful = sum(status.values())
        total = len(status)
        
        print(f"Model loading complete: {successful}/{total} successful")
        
        for model_name, success in status.items():
            status_text = "✅" if success else "❌"
            print(f"  {model_name.upper()}: {status_text}")
        
        return status
    
    def get_models(self) -> Tuple[Optional[YOLO], Optional[Any], Optional[Any], Optional[Any]]:
        """
        Get all loaded models
        
        Returns:
            Tuple of (yolo_model, clip_model, clip_preprocess, gemini_model)
        """
        return (
            self.yolo_model,
            self.clip_model, 
            self.clip_preprocess,
            self.gemini_model
        )
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is loaded
        
        Args:
            model_name: Name of model ('yolo', 'clip', 'gemini')
            
        Returns:
            bool: True if model is loaded
        """
        model_map = {
            'yolo': self.yolo_model,
            'clip': self.clip_model,
            'gemini': self.gemini_model
        }
        
        return model_map.get(model_name.lower()) is not None
    
    def get_model_status(self) -> dict:
        """
        Get current status of all models
        
        Returns:
            dict: Status information for each model
        """
        return {
            'yolo_loaded': self.yolo_model is not None,
            'clip_loaded': self.clip_model is not None,
            'gemini_loaded': self.gemini_model is not None,
            'device': self.device,
            'class_names_count': len(CLASS_NAMES)
        }
    
    def unload_models(self) -> None:
        """Unload all models to free memory"""
        print("Unloading models...")
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Reset model references
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.gemini_model = None
        
        print("Models unloaded successfully")

# Global model loader instance
model_loader = ModelLoader()