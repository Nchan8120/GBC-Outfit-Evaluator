"""
Main outfit analysis module that combines detection, color analysis, and scoring
"""

import cv2
import numpy as np
import torch
import clip
from PIL import Image
from typing import Dict, List, Optional
import time

from app.config import CLASS_NAMES, OCCASIONS, SCORING_WEIGHTS
from app.models.color_detector import ColorDetector
from app.services.model_loader import model_loader

class OutfitAnalyzer:
    """Main class for analyzing outfit images and providing style scores"""
    
    def __init__(self):
        """Initialize outfit analyzer"""
        self.color_detector = ColorDetector()
        self.class_names = CLASS_NAMES
        self.occasions = OCCASIONS
        self.scoring_weights = SCORING_WEIGHTS
        
        # Models will be loaded from model_loader service
        self._models_ready = False
        
        print("OutfitAnalyzer initialized")
    
    def _ensure_models_loaded(self) -> bool:
        """Ensure all required models are loaded"""
        if not self._models_ready:
            yolo_model, clip_model, clip_preprocess, _ = model_loader.get_models()
            
            if yolo_model is None or clip_model is None:
                print("Error: Required models not loaded")
                return False
            
            self._models_ready = True
        
        return True
    
    def analyze_outfit(self, image_path: str, occasion: str) -> Dict:
        """
        Perform complete outfit analysis
        
        Args:
            image_path: Path to the outfit image
            occasion: Occasion type (must be in OCCASIONS keys)
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        # Validate inputs
        if not self._ensure_models_loaded():
            raise RuntimeError("Models not properly loaded")
        
        if occasion not in self.occasions:
            raise ValueError(f"Invalid occasion. Must be one of: {list(self.occasions.keys())}")
        
        print(f"Analyzing outfit for occasion: {occasion}")
        
        # Load and preprocess image
        image = self._load_image(image_path)
        
        # Step 1: Detect clothing items
        detections = self._detect_clothing_items(image_path)
        print(f"Detected {len(detections)} clothing items")
        
        # Step 2: Extract colors for each item
        all_colors = self._extract_colors_from_detections(image, detections)
        
        # Step 3: Calculate different scoring components
        scores = self._calculate_all_scores(image_path, occasion, detections, all_colors)
        
        # Step 4: Calculate final weighted score
        final_score = self._calculate_final_score(scores)
        
        # Step 5: Generate contextual feedback
        feedback = self._generate_feedback(final_score, occasion)
        
        # Step 6: Compile results
        analysis_time = time.time() - start_time
        
        result = {
            'style_score': round(final_score, 1),
            'occasion': occasion,
            'occasion_description': self.occasions[occasion],
            'detected_items': detections,
            'scoring_breakdown': {
                'clip_contextual': round(scores['clip_score'], 1),
                'color_harmony': round(scores['color_harmony'], 1),
                'item_completeness': round(scores['completeness'], 1),
                'style_coherence': round(scores['coherence'], 1)
            },
            'contextual_feedback': feedback,
            'total_items': len(detections),
            'unique_colors': len(set(color['name'] for color in all_colors)),
            'analysis_time_seconds': round(analysis_time, 2)
        }
        
        print(f"Analysis complete in {analysis_time:.2f}s. Final score: {final_score:.1f}/10")
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to RGB format"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def _detect_clothing_items(self, image_path: str) -> List[Dict]:
        """Detect clothing items using YOLO model"""
        yolo_model, _, _, _ = model_loader.get_models()
        
        if yolo_model is None:
            raise RuntimeError("YOLO model not available")
        
        results = yolo_model(image_path)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'class': self.class_names[class_id],
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    detections.append(detection)
        
        return detections
    
    def _extract_colors_from_detections(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Extract colors from all detected clothing items"""
        all_colors = []
        
        print("Extracting colors from detected items...")
        
        for i, detection in enumerate(detections):
            print(f"  Processing item {i+1}: {detection['class']}")
            
            colors = self.color_detector.get_colors_from_bbox(
                image, 
                detection['bbox'], 
                n_colors=2
            )
            
            detection['colors'] = colors
            all_colors.extend(colors)
            
            color_names = [c['name'] for c in colors]
            print(f"    Colors found: {color_names}")
        
        return all_colors
    
    def _calculate_all_scores(self, image_path: str, occasion: str, 
                            detections: List[Dict], all_colors: List[Dict]) -> Dict:
        """Calculate all scoring components"""
        print("Calculating scores...")
        
        scores = {}
        
        # CLIP contextual score
        scores['clip_score'] = self._calculate_clip_score(image_path, occasion)
        print(f"  CLIP score: {scores['clip_score']:.1f}/10")
        
        # Color harmony score
        scores['color_harmony'] = self._calculate_color_harmony_score(all_colors)
        print(f"  Color harmony: {scores['color_harmony']:.1f}/10")
        
        # Item completeness score
        scores['completeness'] = self._calculate_completeness_score(detections, occasion)
        print(f"  Completeness: {scores['completeness']:.1f}/10")
        
        # Style coherence score
        scores['coherence'] = self._calculate_coherence_score(detections, occasion)
        print(f"  Coherence: {scores['coherence']:.1f}/10")
        
        return scores
    
    def _calculate_clip_score(self, image_path: str, occasion: str) -> float:
        """Calculate CLIP-based contextual appropriateness score"""
        _, clip_model, clip_preprocess, _ = model_loader.get_models()
        
        if clip_model is None or clip_preprocess is None:
            print("Warning: CLIP model not available, using fallback score")
            return 6.0
        
        try:
            # Load and preprocess image for CLIP
            image = Image.open(image_path)
            image_input = clip_preprocess(image).unsqueeze(0).to(model_loader.device)
            
            # Get occasion context
            occasion_context = self.occasions[occasion]
            
            # Create context-specific prompts
            prompts = [
                f"professional outfit suitable for {occasion_context}",
                f"appropriate and stylish attire for {occasion_context}",
                f"well-dressed and coordinated for {occasion_context}",
                f"fashionable look perfect for {occasion_context}"
            ]
            
            # Calculate similarities
            similarities = []
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                for prompt in prompts:
                    text_tokens = clip.tokenize([prompt]).to(model_loader.device)
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (image_features @ text_features.T).item()
                    similarities.append(similarity)
            
            # Convert best similarity to 0-10 scale
            best_similarity = max(similarities)
            clip_score = (best_similarity + 1) / 2 * 10
            
            return min(max(clip_score, 0), 10)
            
        except Exception as e:
            print(f"Error in CLIP scoring: {e}")
            return 6.0
    
    def _calculate_color_harmony_score(self, all_colors: List[Dict]) -> float:
        """Calculate color harmony score based on color theory"""
        if not all_colors:
            return 5.0
        
        color_names = [color['name'] for color in all_colors]
        unique_colors = list(set(color_names))
        
        score = 7.0
        
        # Penalty for too many different colors
        if len(unique_colors) > 4:
            score -= 2.0
        elif len(unique_colors) > 3:
            score -= 1.0
        
        # Bonus for having neutrals (versatile)
        neutral_colors = {'black', 'white', 'gray', 'dark_gray', 'light_gray', 'beige'}
        if any(color in neutral_colors for color in unique_colors):
            score += 1.0
        
        # Check for potential clashing
        if self._has_clashing_colors(unique_colors):
            score -= 1.5
        
        return min(max(score, 0), 10)
    
    def _has_clashing_colors(self, color_names: List[str]) -> bool:
        """Check for potentially clashing color combinations"""
        # Simple clash detection
        warm_colors = {'red', 'orange', 'yellow', 'pink'}
        cool_colors = {'blue', 'green', 'purple', 'teal', 'navy'}
        
        warm_count = sum(1 for color in color_names if color in warm_colors)
        cool_count = sum(1 for color in color_names if color in cool_colors)
        
        # Too many warm and cool colors mixed
        return warm_count > 1 and cool_count > 1
    
    def _calculate_completeness_score(self, detections: List[Dict], occasion: str) -> float:
        """Score based on having appropriate items for the occasion"""
        items = [d['class'] for d in detections]
        score = 5.0
        
        if occasion in ['job_interview', 'work_meeting', 'business_casual']:
            # Formal occasions
            if 'shirt' in items and ('pants' in items or 'skirt' in items):
                score += 2.0
            if 'jacket' in items:
                score += 1.0
            if 'shoe' in items:
                score += 1.0
            if 'shorts' in items:
                score -= 2.0  # Too casual
                
        elif occasion in ['formal_event']:
            # Very formal
            if 'dress' in items or ('shirt' in items and 'pants' in items):
                score += 2.0
            if 'shoe' in items:
                score += 1.0
            if 'shorts' in items or 'sunglass' in items:
                score -= 1.0
                
        elif occasion in ['beach_vacation']:
            # Casual/beach appropriate
            if 'shorts' in items or 'skirt' in items or 'dress' in items:
                score += 1.0
            if 'sunglass' in items:
                score += 1.0
            if 'jacket' in items:
                score -= 1.0  # Too formal for beach
                
        else:
            # General casual occasions
            if len(items) >= 2:
                score += 1.0
            if 'shoe' in items:
                score += 0.5
        
        return min(max(score, 0), 10)
    
    def _calculate_coherence_score(self, detections: List[Dict], occasion: str) -> float:
        """Score based on style consistency"""
        items = [d['class'] for d in detections]
        score = 7.0
        
        formal_items = ['jacket', 'shirt']
        casual_items = ['shorts', 'sunglass']
        
        has_formal = any(item in items for item in formal_items)
        has_casual = any(item in items for item in casual_items)
        
        # Check for style mismatches
        if occasion in ['job_interview', 'work_meeting', 'formal_event']:
            if has_casual and has_formal:
                score -= 2.0  # Mixed formality
            elif has_casual:
                score -= 3.0  # Too casual
        elif occasion in ['beach_vacation', 'casual_hangout']:
            if has_formal and not has_casual:
                score -= 1.0  # Potentially overdressed
        
        return min(max(score, 0), 10)
    
    def _calculate_final_score(self, scores: Dict) -> float:
        """Calculate weighted final score"""
        final_score = (
            self.scoring_weights['clip_contextual'] * scores['clip_score'] +
            self.scoring_weights['color_harmony'] * scores['color_harmony'] +
            self.scoring_weights['item_completeness'] * scores['completeness'] +
            self.scoring_weights['style_coherence'] * scores['coherence']
        )
        
        return min(max(final_score, 0), 10)
    
    def _generate_feedback(self, final_score: float, occasion: str) -> str:
        """Generate contextual feedback based on score and occasion"""
        occasion_name = self.occasions[occasion]
        
        if final_score >= 8:
            return f"Excellent choice for {occasion_name}! Very well put together."
        elif final_score >= 6:
            return f"Good outfit for {occasion_name}. Well coordinated overall."
        elif final_score >= 4:
            return f"Decent look for {occasion_name}, but could use some improvements."
        else:
            return f"This outfit may not be the best choice for {occasion_name}."
    
    def cleanup(self):
        """Clean up resources"""
        self.color_detector.cleanup_all_temp_files()