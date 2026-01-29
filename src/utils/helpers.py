"""
Helper utility functions
"""

import os
import hashlib
from pathlib import Path
from typing import Optional


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: str) -> str:
    """
    Get file extension
    
    Args:
        file_path: File path
        
    Returns:
        File extension (e.g., '.mp3')
    """
    return Path(file_path).suffix.lower()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to HH:MM:SS
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def cleanup_file(file_path: str) -> bool:
    """
    Delete file if exists
    
    Args:
        file_path: Path to file
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception:
        pass
    return False


def extract_s3_key_from_url(s3_url: str) -> Optional[str]:
    """
    Extract S3 key from S3 URL
    
    Args:
        s3_url: S3 URL (e.g., https://bucket.s3.region.amazonaws.com/key/file.mp3)
        
    Returns:
        S3 key or None
    """
    try:
        # Remove protocol
        url = s3_url.replace('https://', '').replace('http://', '')
        
        # Split by first /
        parts = url.split('/', 1)
        if len(parts) == 2:
            return parts[1]
    except Exception:
        pass
    
    return None
