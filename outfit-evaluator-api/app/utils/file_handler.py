"""
File handling utilities for image uploads and processing
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import uuid
from PIL import Image
import mimetypes

from app.config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES

class FileHandler:
    """Handles file upload, validation, and cleanup operations"""
    
    def __init__(self):
        """Initialize file handler"""
        self.upload_dir = Path(UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
        print(f"FileHandler initialized. Upload directory: {self.upload_dir}")
    
    def validate_file(self, file_content: bytes, filename: str, content_type: str) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME content type
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            size_mb = len(file_content) / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large: {size_mb:.1f}MB. Maximum allowed: {max_mb}MB"
        
        # Check file extension
        if not filename:
            return False, "No filename provided"
        
        file_extension = Path(filename).suffix.lower().lstrip('.')
        if file_extension not in ALLOWED_EXTENSIONS:
            return False, f"Invalid file type: .{file_extension}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        
        # Check MIME type
        if content_type not in ALLOWED_MIME_TYPES:
            return False, f"Invalid content type: {content_type}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        
        # Try to open as image to verify it's a valid image
        try:
            from io import BytesIO
            image = Image.open(BytesIO(file_content))
            image.verify()  # Verify it's a valid image
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
        
        return True, "Valid file"
    
    def save_upload(self, file_content: bytes, original_filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Save uploaded file with unique filename
        
        Args:
            file_content: File content as bytes
            original_filename: Original filename
            
        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Generate unique filename
            unique_filename = self._generate_unique_filename(original_filename)
            file_path = self.upload_dir / unique_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            print(f"File saved: {file_path}")
            return True, "File saved successfully", str(file_path)
            
        except Exception as e:
            return False, f"Error saving file: {str(e)}", None
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename with timestamp and UUID"""
        
        # Get file extension
        file_extension = Path(original_filename).suffix.lower()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate short UUID
        unique_id = str(uuid.uuid4())[:8]
        
        # Combine for unique filename
        unique_filename = f"{timestamp}_{unique_id}{file_extension}"
        
        return unique_filename
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up a specific file
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
                return True
            return False
            
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old files from upload directory
        
        Args:
            max_age_hours: Maximum age in hours before deletion
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        current_time = datetime.now()
        
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    # Get file modification time
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_hours = (current_time - file_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        if self.cleanup_file(str(file_path)):
                            deleted_count += 1
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old files")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return deleted_count
    
    def get_file_info(self, file_path: str) -> Optional[dict]:
        """
        Get information about a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information or None
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            file_stat = os.stat(file_path)
            file_path_obj = Path(file_path)
            
            # Get image dimensions if it's an image
            dimensions = None
            try:
                with Image.open(file_path) as img:
                    dimensions = {"width": img.width, "height": img.height}
            except:
                pass
            
            return {
                "filename": file_path_obj.name,
                "size_bytes": file_stat.st_size,
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "extension": file_path_obj.suffix.lower(),
                "mime_type": mimetypes.guess_type(file_path)[0],
                "dimensions": dimensions
            }
            
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return None
    
    def optimize_image(self, file_path: str, max_width: int = 1024, max_height: int = 1024, quality: int = 85) -> bool:
        """
        Optimize image file by resizing and compressing
        
        Args:
            file_path: Path to image file
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            quality: JPEG quality (1-100)
            
        Returns:
            bool: True if successful
        """
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate new dimensions while maintaining aspect ratio
                original_width, original_height = img.size
                
                if original_width <= max_width and original_height <= max_height:
                    # Image is already small enough
                    return True
                
                # Calculate scaling factor
                width_ratio = max_width / original_width
                height_ratio = max_height / original_height
                scale_factor = min(width_ratio, height_ratio)
                
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Resize image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save optimized image
                resized_img.save(file_path, 'JPEG', quality=quality, optimize=True)
                
                print(f"Image optimized: {original_width}x{original_height} -> {new_width}x{new_height}")
                return True
                
        except Exception as e:
            print(f"Error optimizing image {file_path}: {e}")
            return False
    
    def get_upload_stats(self) -> dict:
        """
        Get statistics about the upload directory
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            file_count = 0
            total_size = 0
            file_types = {}
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_count += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    extension = file_path.suffix.lower()
                    file_types[extension] = file_types.get(extension, 0) + 1
            
            return {
                "total_files": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "upload_directory": str(self.upload_dir)
            }
            
        except Exception as e:
            print(f"Error getting upload stats: {e}")
            return {"error": str(e)}

# Global file handler instance
file_handler = FileHandler()