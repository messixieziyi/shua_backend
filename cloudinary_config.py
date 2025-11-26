"""
Cloudinary Configuration and Utilities
======================================

This module provides utilities for uploading and managing images on Cloudinary.
Images are uploaded to Cloudinary and URLs are returned for storage in the database.
"""

import os
import base64
import re
from typing import Optional, Tuple
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.exceptions import Error as CloudinaryError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Cloudinary with environment variables
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def is_cloudinary_configured() -> bool:
    """Check if Cloudinary is properly configured."""
    return all([
        os.getenv("CLOUDINARY_CLOUD_NAME"),
        os.getenv("CLOUDINARY_API_KEY"),
        os.getenv("CLOUDINARY_API_SECRET")
    ])

def extract_public_id_from_url(url: str) -> Optional[str]:
    """
    Extract the public_id from a Cloudinary URL.
    
    Example:
        https://res.cloudinary.com/demo/image/upload/v1234567890/sample.jpg
        -> sample
    """
    try:
        # Match Cloudinary URL pattern
        match = re.search(r'/upload/(?:v\d+/)?(.+?)(?:\.[^.]+)?$', url)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None

def upload_image(
    image_data: str,
    folder: str = "shua",
    max_size_mb: int = 5,
    resource_type: str = "image"
) -> Tuple[bool, str]:
    """
    Upload an image to Cloudinary.
    
    Args:
        image_data: Base64 encoded image data (data:image/...;base64,...) or HTTP(S) URL
        folder: Cloudinary folder to organize images
        max_size_mb: Maximum allowed image size in MB
        resource_type: Type of resource (default: "image")
    
    Returns:
        Tuple of (success: bool, result: str)
        - If success: (True, cloudinary_url)
        - If failure: (False, error_message)
    """
    try:
        # If it's already a Cloudinary URL, return it as-is
        if image_data.startswith('https://res.cloudinary.com/'):
            return True, image_data
        
        # If it's an HTTP/HTTPS URL (not Cloudinary), return it as-is for now
        # This handles existing Unsplash URLs or other external images
        if image_data.startswith('http://') or image_data.startswith('https://'):
            return True, image_data
        
        # Check if Cloudinary is configured
        if not is_cloudinary_configured():
            return False, "Cloudinary is not configured. Please set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET environment variables."
        
        # Validate base64 data URL format
        if not image_data.startswith('data:image/'):
            return False, "Invalid image format. Must be a data URL (data:image/...)."
        
        # Extract and validate MIME type
        try:
            header, data = image_data.split(',', 1)
        except ValueError:
            return False, "Invalid data URL format."
        
        mime_match = re.match(r'data:image/(jpeg|jpg|png|gif|webp)', header)
        if not mime_match:
            return False, "Unsupported image type. Only JPEG, PNG, GIF, and WebP are allowed."
        
        # Validate size
        try:
            decoded = base64.b64decode(data)
            size_mb = len(decoded) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                return False, f"Image too large. Maximum size is {max_size_mb}MB."
        except Exception:
            return False, "Invalid base64 encoding."
        
        # Upload to Cloudinary
        try:
            result = cloudinary.uploader.upload(
                image_data,
                folder=folder,
                resource_type=resource_type,
                # Generate unique filename
                use_filename=False,
                unique_filename=True,
                # Optimize for web delivery
                quality="auto",
                fetch_format="auto"
            )
            
            # Return the secure URL
            return True, result.get('secure_url', result.get('url'))
            
        except CloudinaryError as e:
            return False, f"Cloudinary upload failed: {str(e)}"
            
    except Exception as e:
        return False, f"Image upload error: {str(e)}"

def delete_image(image_url: str) -> Tuple[bool, str]:
    """
    Delete an image from Cloudinary.
    
    Args:
        image_url: Cloudinary URL of the image to delete
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Only delete if it's a Cloudinary URL
        if not image_url.startswith('https://res.cloudinary.com/'):
            return True, "Not a Cloudinary URL, skipping deletion."
        
        # Check if Cloudinary is configured
        if not is_cloudinary_configured():
            return False, "Cloudinary is not configured."
        
        # Extract public_id from URL
        public_id = extract_public_id_from_url(image_url)
        if not public_id:
            return False, "Could not extract public_id from URL."
        
        # Delete from Cloudinary
        try:
            result = cloudinary.uploader.destroy(public_id)
            
            if result.get('result') == 'ok':
                return True, "Image deleted successfully."
            else:
                return False, f"Deletion failed: {result.get('result', 'unknown error')}"
                
        except CloudinaryError as e:
            return False, f"Cloudinary deletion failed: {str(e)}"
            
    except Exception as e:
        return False, f"Image deletion error: {str(e)}"

def validate_and_upload_image(
    image_data: Optional[str],
    old_image_url: Optional[str] = None,
    folder: str = "shua",
    max_size_mb: int = 5
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and upload an image, optionally deleting the old one.
    
    Args:
        image_data: New image data (base64 or URL) or None
        old_image_url: URL of the old image to delete (if replacing)
        folder: Cloudinary folder
        max_size_mb: Maximum size in MB
    
    Returns:
        Tuple of (success: bool, message: str, new_url: Optional[str])
    """
    try:
        # If no new image data, return None
        if image_data is None:
            return True, "No image provided", None
        
        # If empty string, delete old image and return None
        if image_data == "":
            if old_image_url:
                delete_image(old_image_url)
            return True, "Image removed", None
        
        # Upload new image
        success, result = upload_image(image_data, folder=folder, max_size_mb=max_size_mb)
        
        if not success:
            return False, result, None
        
        new_url = result
        
        # Delete old image if it exists and is different from new one
        if old_image_url and old_image_url != new_url:
            delete_image(old_image_url)
        
        return True, "Image uploaded successfully", new_url
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None
