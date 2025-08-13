"""
Improved color detection module with better color classification
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Tuple, Optional
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
        try:
            print(f"  🎨 Extracting colors from bbox: {bbox}")
            
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                print(f"    ❌ Invalid bbox dimensions")
                return []
            
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print(f"    ❌ Empty cropped region")
                return []
            
            print(f"    📐 Analyzing region: {x2-x1}x{y2-y1} pixels")
            
            # Method 1: Simple color extraction (fallback)
            colors_simple = self._extract_simple_colors(cropped, n_colors)
            
            # Method 2: Try ColorThief if available
            colors_colorthief = []
            try:
                colors_colorthief = self._extract_with_colorthief(cropped)
            except Exception as e:
                print(f"    ⚠️ ColorThief failed: {e}")
            
            # Method 3: HSV analysis (most reliable)
            colors_hsv = self._extract_with_hsv_analysis(cropped, n_colors)
            
            # Combine results
            combined_colors = self._combine_color_results(
                colors_simple, colors_colorthief, colors_hsv, n_colors
            )
            
            print(f"    ✅ Found {len(combined_colors)} colors: {[c['name'] for c in combined_colors]}")
            return combined_colors[:n_colors]
            
        except Exception as e:
            print(f"    ❌ Color extraction error: {e}")
            return self._get_fallback_colors()
    
    def _extract_simple_colors(self, cropped_region: np.ndarray, n_colors: int) -> List[Dict]:
        """Simple color extraction using basic statistics"""
        try:
            # Reshape to list of pixels
            pixels = cropped_region.reshape(-1, 3)
            
            # Remove extreme values (very dark/bright)
            mask = np.all(pixels > 20, axis=1) & np.all(pixels < 240, axis=1)
            if np.sum(mask) < 10:  # Fallback if too few pixels
                mask = np.ones(len(pixels), dtype=bool)
            
            valid_pixels = pixels[mask]
            
            if len(valid_pixels) == 0:
                return []
            
            # Get most common colors using simple binning
            colors = []
            
            # Method: Average color
            avg_color = np.mean(valid_pixels, axis=0).astype(int)
            color_name = self._classify_color_simple(tuple(avg_color))
            colors.append({
                'rgb': avg_color.tolist(),
                'name': color_name,
                'method': 'average'
            })
            
            # Method: Most saturated color
            hsv_pixels = np.array([cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2HSV)[0][0] 
                                  for pixel in valid_pixels[:100:5]])  # Sample for speed
            
            if len(hsv_pixels) > 0:
                most_saturated_idx = np.argmax(hsv_pixels[:, 1])  # Highest saturation
                most_saturated_rgb = valid_pixels[most_saturated_idx * 5]
                color_name = self._classify_color_simple(tuple(most_saturated_rgb))
                
                if color_name != colors[0]['name']:  # Avoid duplicates
                    colors.append({
                        'rgb': most_saturated_rgb.tolist(),
                        'name': color_name,
                        'method': 'saturated'
                    })
            
            return colors
            
        except Exception as e:
            print(f"    ⚠️ Simple extraction failed: {e}")
            return []
    
    def _extract_with_colorthief(self, cropped_region: np.ndarray) -> List[Dict]:
        """Extract colors using ColorThief (if available)"""
        temp_path = None
        try:
            # Save temporary file
            temp_path = self._save_temp_image(cropped_region)
            
            color_thief = ColorThief(temp_path)
            palette = color_thief.get_palette(color_count=3, quality=10)
            
            colors = []
            for rgb in palette:
                color_name = self._classify_color_simple(rgb)
                colors.append({
                    'rgb': list(rgb),
                    'name': color_name,
                    'method': 'colorthief'
                })
            
            return colors
            
        except Exception as e:
            print(f"    ⚠️ ColorThief error: {e}")
            return []
        finally:
            if temp_path:
                self._cleanup_temp_file(temp_path)
    
    def _extract_with_hsv_analysis(self, cropped_region: np.ndarray, n_colors: int) -> List[Dict]:
        """Extract colors using HSV analysis"""
        try:
            # Convert to HSV
            hsv_image = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2HSV)
            
            # Create mask for valid pixels
            mask = self._create_valid_pixel_mask(hsv_image)
            
            if np.sum(mask) < 50:  # Not enough valid pixels
                print(f"    ⚠️ Too few valid pixels for HSV analysis")
                return []
            
            # Get valid pixels
            valid_pixels_rgb = cropped_region[mask]
            valid_pixels_hsv = hsv_image[mask]
            
            # Try K-means clustering if sklearn is available
            try:
                from sklearn.cluster import KMeans
                
                n_clusters = min(n_colors, max(1, len(valid_pixels_rgb) // 100))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(valid_pixels_rgb)
                
                colors = []
                labels = kmeans.labels_
                
                for i, center_rgb in enumerate(kmeans.cluster_centers_):
                    # Count pixels in this cluster
                    cluster_size = np.sum(labels == i)
                    cluster_percentage = (cluster_size / len(labels)) * 100
                    
                    # Skip very small clusters
                    if cluster_percentage < 15:
                        continue
                    
                    color_name = self._classify_color_simple(tuple(center_rgb.astype(int)))
                    
                    colors.append({
                        'rgb': center_rgb.astype(int).tolist(),
                        'name': color_name,
                        'method': 'kmeans',
                        'percentage': cluster_percentage
                    })
                
                # Sort by cluster size
                colors.sort(key=lambda x: x.get('percentage', 0), reverse=True)
                return colors
                
            except ImportError:
                print(f"    ⚠️ sklearn not available, using simple HSV analysis")
                # Fallback: just get dominant colors by averaging
                return self._simple_hsv_analysis(valid_pixels_rgb, valid_pixels_hsv)
                
        except Exception as e:
            print(f"    ⚠️ HSV analysis error: {e}")
            return []
    
    def _simple_hsv_analysis(self, valid_pixels_rgb: np.ndarray, valid_pixels_hsv: np.ndarray) -> List[Dict]:
        """Simple HSV analysis without clustering"""
        colors = []
        
        # Get average color
        avg_rgb = np.mean(valid_pixels_rgb, axis=0).astype(int)
        color_name = self._classify_color_simple(tuple(avg_rgb))
        colors.append({
            'rgb': avg_rgb.tolist(),
            'name': color_name,
            'method': 'hsv_average'
        })
        
        # Get most saturated color
        if len(valid_pixels_hsv) > 0:
            max_sat_idx = np.argmax(valid_pixels_hsv[:, 1])
            saturated_rgb = valid_pixels_rgb[max_sat_idx]
            color_name = self._classify_color_simple(tuple(saturated_rgb))
            
            if color_name != colors[0]['name']:
                colors.append({
                    'rgb': saturated_rgb.tolist(),
                    'name': color_name,
                    'method': 'hsv_saturated'
                })
        
        return colors
    
    def _create_valid_pixel_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create mask for valid pixels (not shadows/highlights)"""
        h, s, v = cv2.split(hsv_image)
        
        # Remove very dark pixels (shadows)
        mask_dark = v > 25
        
        # Remove very bright pixels (highlights/overexposure)
        mask_bright = v < 245
        
        # Remove very desaturated pixels in bright areas
        mask_desat = (s > 10) | (v < 200)
        
        return mask_dark & mask_bright & mask_desat
    
    def _classify_color_simple(self, rgb: Tuple[int, int, int]) -> str:
        """Simplified color classification"""
        r, g, b = rgb
        
        # Handle edge cases
        if any(c < 0 or c > 255 for c in rgb):
            return 'unknown'
        
        try:
            # Convert to HSV for better classification
            hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv
            
            # Grayscale (low saturation)
            if s < 30:
                if v < 50:
                    return 'black'
                elif v < 100:
                    return 'gray'
                elif v < 200:
                    return 'light_gray'
                else:
                    return 'white'
            
            # Dark colors
            if v < 60:
                return 'black'
            
            # Classify by hue
            if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
                return 'red'
            elif h > 10 and h <= 25:
                return 'orange'
            elif h > 25 and h <= 35:
                return 'yellow'
            elif h > 35 and h <= 85:
                return 'green'
            elif h > 85 and h <= 95:
                return 'cyan'
            elif h > 95 and h <= 125:
                return 'blue'
            elif h > 125 and h <= 145:
                return 'purple'
            elif h > 145 and h < 170:
                return 'pink'
            
            return 'unknown'
            
        except Exception as e:
            print(f"    ⚠️ Color classification error: {e}")
            return 'unknown'
    
    def _combine_color_results(self, colors1: List[Dict], colors2: List[Dict], 
                             colors3: List[Dict], n_colors: int) -> List[Dict]:
        """Combine results from different methods and remove duplicates"""
        all_colors = colors1 + colors2 + colors3
        
        # Remove duplicates and unknowns
        seen_colors = set()
        unique_colors = []
        
        for color in all_colors:
            color_name = color['name']
            if color_name not in seen_colors and color_name != 'unknown':
                seen_colors.add(color_name)
                unique_colors.append(color)
        
        # Sort by preference: kmeans > hsv > colorthief > simple
        method_priority = {
            'kmeans': 1,
            'hsv_average': 2,
            'hsv_saturated': 3,
            'colorthief': 4,
            'saturated': 5,
            'average': 6
        }
        
        unique_colors.sort(key=lambda x: (
            method_priority.get(x.get('method', 'unknown'), 99),
            -x.get('percentage', 0)
        ))
        
        return unique_colors
    
    def _get_fallback_colors(self) -> List[Dict]:
        """Fallback colors when detection fails"""
        return [
            {'rgb': [128, 128, 128], 'name': 'gray', 'method': 'fallback'},
            {'rgb': [64, 64, 64], 'name': 'dark_gray', 'method': 'fallback'}
        ]
    
    def _save_temp_image(self, image_region: np.ndarray) -> str:
        """Save image region to temporary file"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd)
            
            # Ensure image is valid
            if image_region.size == 0:
                raise ValueError("Empty image region")
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_region, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(temp_path, image_bgr)
            
            if not success:
                raise ValueError("Failed to write image")
            
            self.temp_files.append(temp_path)
            return temp_path
            
        except Exception as e:
            print(f"    ⚠️ Failed to save temp image: {e}")
            raise
    
    def _cleanup_temp_file(self, temp_path: str) -> None:
        """Clean up a specific temporary file"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
        except Exception as e:
            print(f"⚠️ Could not clean up temp file {temp_path}: {e}")
    
    def cleanup_all_temp_files(self) -> None:
        """Clean up all temporary files"""
        for temp_path in self.temp_files.copy():
            self._cleanup_temp_file(temp_path)
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        try:
            self.cleanup_all_temp_files()
        except:
            pass  # Ignore errors during cleanup