"""
Improved color detection module with better color classification
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Tuple
from colorthief import ColorThief
import colorsys

class ColorDetector:
    """Improved color detector with better color classification"""
    
    def __init__(self):
        """Initialize color detector with comprehensive color mappings"""
        self.temp_files = []
        
        # Define color ranges in HSV for better detection
        self.color_ranges = {
            # Hue ranges (0-180 in OpenCV)
            'red': [(0, 10), (170, 180)],  # Red wraps around
            'orange': [(10, 25)],
            'yellow': [(25, 35)],
            'green': [(35, 85)],
            'cyan': [(85, 95)],
            'blue': [(95, 125)],
            'purple': [(125, 145)],
            'pink': [(145, 170)],
        }
        
        # Saturation and Value thresholds
        self.low_saturation_threshold = 30  # Below this = grayscale/neutral
        self.low_value_threshold = 40       # Below this = dark colors
        self.high_value_threshold = 200     # Above this = light colors
    
    def get_colors_from_bbox(self, image: np.ndarray, bbox: List[int], n_colors: int = 2) -> List[Dict]:
        """Extract colors with improved classification"""
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return []
        
        print(f"    Analyzing region: {x2-x1}x{y2-y1} pixels")
        
        # Method 1: Use ColorThief for initial color extraction
        colors_colorthief = self._extract_with_colorthief(cropped)
        
        # Method 2: Use HSV analysis for better classification
        colors_hsv = self._extract_with_hsv_analysis(cropped, n_colors)
        
        # Combine and deduplicate results
        combined_colors = self._combine_color_results(colors_colorthief, colors_hsv, n_colors)
        
        return combined_colors[:n_colors]
    
    def _extract_with_colorthief(self, cropped_region: np.ndarray) -> List[Dict]:
        """Extract colors using ColorThief"""
        # Save temporary file
        temp_path = self._save_temp_image(cropped_region)
        
        try:
            color_thief = ColorThief(temp_path)
            palette = color_thief.get_palette(color_count=5, quality=1)
            
            colors = []
            for rgb in palette:
                color_name = self._classify_color_advanced(rgb)
                colors.append({
                    'rgb': list(rgb),
                    'name': color_name,
                    'method': 'colorthief'
                })
            
            return colors
            
        except Exception as e:
            print(f"    ColorThief error: {e}")
            return []
        finally:
            self._cleanup_temp_file(temp_path)
    
    def _extract_with_hsv_analysis(self, cropped_region: np.ndarray, n_colors: int) -> List[Dict]:
        """Extract colors using HSV analysis"""
        # Convert to HSV
        hsv_image = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2HSV)
        
        # Remove very dark and very bright pixels (shadows/highlights)
        mask = self._create_valid_pixel_mask(hsv_image)
        
        if np.sum(mask) < 100:  # Not enough valid pixels
            return []
        
        # Get valid pixels
        valid_pixels_rgb = cropped_region[mask]
        valid_pixels_hsv = hsv_image[mask]
        
        # Use K-means clustering on valid pixels
        from sklearn.cluster import KMeans
        
        n_clusters = min(n_colors, len(valid_pixels_rgb) // 50)  # At least 50 pixels per cluster
        if n_clusters < 1:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(valid_pixels_rgb)
        
        colors = []
        labels = kmeans.labels_
        
        for i, center_rgb in enumerate(kmeans.cluster_centers_):
            # Get corresponding HSV for this cluster
            center_hsv = cv2.cvtColor(center_rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
            
            # Count pixels in this cluster
            cluster_size = np.sum(labels == i)
            cluster_percentage = (cluster_size / len(labels)) * 100
            
            # Skip very small clusters
            if cluster_percentage < 10:
                continue
            
            color_name = self._classify_color_from_hsv(center_hsv, center_rgb.astype(int))
            
            colors.append({
                'rgb': center_rgb.astype(int).tolist(),
                'name': color_name,
                'method': 'hsv_analysis',
                'percentage': cluster_percentage
            })
        
        # Sort by cluster size
        colors.sort(key=lambda x: x.get('percentage', 0), reverse=True)
        
        return colors
    
    def _create_valid_pixel_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create mask for valid pixels (not shadows/highlights)"""
        h, s, v = cv2.split(hsv_image)
        
        # Remove very dark pixels (shadows)
        mask_dark = v > 30
        
        # Remove very bright pixels (highlights/overexposure)
        mask_bright = v < 240
        
        # Remove very desaturated pixels in bright areas (often white/gray backgrounds)
        mask_desat = (s > 15) | (v < 180)
        
        return mask_dark & mask_bright & mask_desat
    
    def _classify_color_advanced(self, rgb: Tuple[int, int, int]) -> str:
        """Advanced color classification using multiple methods"""
        r, g, b = rgb
        
        # Convert to HSV for hue-based classification
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
        
        return self._classify_color_from_hsv(hsv, rgb)
    
    def _classify_color_from_hsv(self, hsv: np.ndarray, rgb: Tuple[int, int, int]) -> str:
        """Classify color using HSV values"""
        h, s, v = hsv
        
        print(f"      HSV: ({h}, {s}, {v}) -> RGB{rgb}")
        
        # Handle grayscale colors (low saturation)
        if s < self.low_saturation_threshold:
            return self._classify_grayscale(v)
        
        # Handle very dark colors
        if v < self.low_value_threshold:
            return self._classify_dark_color(h, s, v)
        
        # Handle very light colors
        if v > self.high_value_threshold and s < 80:
            return self._classify_light_color(h, s, v)
        
        # Classify by hue for saturated colors
        return self._classify_by_hue(h, s, v)
    
    def _classify_grayscale(self, value: int) -> str:
        """Classify grayscale colors based on brightness"""
        if value < 40:
            return 'black'
        elif value < 80:
            return 'dark_gray'
        elif value < 120:
            return 'gray'
        elif value < 160:
            return 'light_gray'
        elif value < 220:
            return 'silver'
        else:
            return 'white'
    
    def _classify_dark_color(self, hue: int, sat: int, val: int) -> str:
        """Classify dark colors"""
        if sat < 30:
            return 'black' if val < 30 else 'dark_gray'
        
        # Dark but saturated colors
        if 95 <= hue <= 125:  # Blue range
            return 'navy'
        elif 35 <= hue <= 85:  # Green range
            return 'dark_green'
        elif (0 <= hue <= 10) or (170 <= hue <= 180):  # Red range
            return 'dark_red'
        else:
            return 'dark_gray'
    
    def _classify_light_color(self, hue: int, sat: int, val: int) -> str:
        """Classify light/pastel colors"""
        if sat < 30:
            return 'white' if val > 230 else 'light_gray'
        
        # Light but with some saturation (pastels)
        if 95 <= hue <= 125:
            return 'light_blue'
        elif 145 <= hue <= 170:
            return 'pink'
        elif 25 <= hue <= 35:
            return 'cream'
        else:
            return 'beige'
    
    def _classify_by_hue(self, hue: int, sat: int, val: int) -> str:
        """Classify color by hue for saturated colors"""
        
        # Red (including wrapping around 180)
        if (0 <= hue <= 10) or (170 <= hue <= 180):
            if val > 150:
                return 'red'
            else:
                return 'dark_red'
        
        # Orange
        elif 10 < hue <= 25:
            return 'orange'
        
        # Yellow
        elif 25 < hue <= 35:
            if sat > 100:
                return 'yellow'
            else:
                return 'cream'
        
        # Green
        elif 35 < hue <= 85:
            if 35 < hue <= 50:
                return 'yellow_green'
            elif 50 < hue <= 70:
                return 'green'
            else:
                return 'teal'
        
        # Cyan
        elif 85 < hue <= 95:
            return 'cyan'
        
        # Blue
        elif 95 < hue <= 125:
            if val < 100:
                return 'navy'
            elif 95 < hue <= 110:
                return 'blue'
            else:
                return 'royal_blue'
        
        # Purple
        elif 125 < hue <= 145:
            return 'purple'
        
        # Pink/Magenta
        elif 145 < hue <= 170:
            if val > 180:
                return 'pink'
            else:
                return 'magenta'
        
        return 'unknown'
    
    def _combine_color_results(self, colors1: List[Dict], colors2: List[Dict], n_colors: int) -> List[Dict]:
        """Combine results from different methods and remove duplicates"""
        all_colors = colors1 + colors2
        
        # Remove duplicates based on color name
        seen_colors = set()
        unique_colors = []
        
        for color in all_colors:
            color_name = color['name']
            if color_name not in seen_colors and color_name != 'unknown':
                seen_colors.add(color_name)
                unique_colors.append(color)
        
        # Sort by preference: HSV analysis first (more accurate), then ColorThief
        unique_colors.sort(key=lambda x: (
            x.get('method') != 'hsv_analysis',  # HSV first
            -x.get('percentage', 0)  # Then by percentage (higher first)
        ))
        
        return unique_colors
    
    def _save_temp_image(self, image_region: np.ndarray) -> str:
        """Save image region to temporary file"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        
        image_bgr = cv2.cvtColor(image_region, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, image_bgr)
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def _cleanup_temp_file(self, temp_path: str) -> None:
        """Clean up a specific temporary file"""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not clean up temp file {temp_path}: {e}")
    
    def cleanup_all_temp_files(self) -> None:
        """Clean up all temporary files"""
        for temp_path in self.temp_files.copy():
            self._cleanup_temp_file(temp_path)
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup_all_temp_files()