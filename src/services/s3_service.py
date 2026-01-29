"""
S3 Service - AWS S3 file operations

Handles downloading audio files from S3 and uploading processed artifacts.
"""

import os
import boto3
from pathlib import Path
from typing import Optional, Tuple
from botocore.exceptions import ClientError, NoCredentialsError

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import S3DownloadError, S3UploadError, ValidationError
from utils.validators import S3URLParser
from utils.helpers import ensure_directory, get_file_extension

logger = get_logger(__name__)


class S3Service:
    """Service for AWS S3 operations"""
    
    def __init__(self):
        """Initialize S3 client"""
        self.bucket_name = settings.AWS_S3_BUCKET
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"✅ S3 client initialized (region: {settings.AWS_REGION})")
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            raise S3DownloadError(
                "AWS credentials not configured",
                details={
                    "hint": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
                }
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            raise S3DownloadError(f"Failed to initialize S3 client: {e}")
    
    def download_file(
        self,
        s3_url: str,
        local_dir: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Download file from S3 to local storage
        
        Args:
            s3_url: S3 URL (s3://bucket/path/to/file.mp3)
            local_dir: Local directory to save file (default: settings.TEMP_DIR)
            filename: Custom filename (default: use original filename)
            
        Returns:
            Local file path
            
        Raises:
            S3DownloadError: If download fails
        """
        try:
            # Parse S3 URL
            bucket_name, object_key = S3URLParser.parse(s3_url)
            
            # Determine local path
            if local_dir is None:
                local_dir = settings.TEMP_DIR
            
            # Ensure directory exists
            ensure_directory(local_dir)
            
            # Determine filename
            if filename is None:
                filename = os.path.basename(object_key)
            
            local_path = os.path.join(local_dir, filename)
            
            logger.info(f"📥 Downloading from S3...")
            logger.info(f"   - Bucket: {bucket_name}")
            logger.info(f"   - Key: {object_key}")
            logger.info(f"   - Local path: {local_path}")
            
            # Check if file exists in S3
            if not self._file_exists(bucket_name, object_key):
                raise S3DownloadError(
                    f"File not found in S3: {object_key}",
                    s3_url=s3_url,
                    details={
                        "bucket": bucket_name,
                        "key": object_key
                    }
                )
            
            # Get file size for progress tracking
            file_size = self._get_file_size(bucket_name, object_key)
            logger.info(f"   - File size: {self._format_size(file_size)}")
            
            # Download file with progress callback
            self.s3_client.download_file(
                Bucket=bucket_name,
                Key=object_key,
                Filename=local_path,
                Callback=ProgressCallback(file_size, object_key)
            )
            
            # Verify downloaded file exists
            if not os.path.exists(local_path):
                raise S3DownloadError(
                    "Downloaded file not found on disk",
                    s3_url=s3_url,
                    details={"expected_path": local_path}
                )
            
            downloaded_size = os.path.getsize(local_path)
            logger.info(f"✅ Download complete: {self._format_size(downloaded_size)}")
            
            # Verify file size matches
            if downloaded_size != file_size:
                logger.warning(
                    f"⚠️ File size mismatch: expected {file_size}, got {downloaded_size}"
                )
            
            return local_path
            
        except S3DownloadError:
            raise
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"❌ S3 download failed: {error_code} - {error_message}")
            
            raise S3DownloadError(
                f"S3 download failed: {error_message}",
                s3_url=s3_url,
                details={
                    "error_code": error_code,
                    "bucket": bucket_name,
                    "key": object_key
                }
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error during S3 download: {e}", exc_info=True)
            raise S3DownloadError(
                f"Unexpected error during download: {str(e)}",
                s3_url=s3_url
            )
    
    def upload_file(
        self,
        local_path: str,
        s3_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        make_public: bool = False
    ) -> str:
        """
        Upload file to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (default: use filename)
            bucket_name: S3 bucket (default: settings.AWS_S3_BUCKET)
            make_public: Make file publicly readable
            
        Returns:
            S3 URL
            
        Raises:
            S3UploadError: If upload fails
        """
        try:
            if not os.path.exists(local_path):
                raise S3UploadError(
                    f"Local file not found: {local_path}",
                    details={"local_path": local_path}
                )
            
            if bucket_name is None:
                bucket_name = self.bucket_name
            
            if s3_key is None:
                s3_key = os.path.basename(local_path)
            
            file_size = os.path.getsize(local_path)
            
            logger.info(f"📤 Uploading to S3...")
            logger.info(f"   - Local: {local_path}")
            logger.info(f"   - Bucket: {bucket_name}")
            logger.info(f"   - Key: {s3_key}")
            logger.info(f"   - Size: {self._format_size(file_size)}")
            
            # Prepare extra args
            extra_args = {}
            if make_public:
                extra_args['ACL'] = 'public-read'
            
            # Upload file
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args if extra_args else None,
                Callback=ProgressCallback(file_size, s3_key)
            )
            
            s3_url = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"✅ Upload complete: {s3_url}")
            
            return s3_url
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"❌ S3 upload failed: {error_code} - {error_message}")
            
            raise S3UploadError(
                f"S3 upload failed: {error_message}",
                details={
                    "error_code": error_code,
                    "bucket": bucket_name,
                    "key": s3_key,
                    "local_path": local_path
                }
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error during S3 upload: {e}", exc_info=True)
            raise S3UploadError(
                f"Unexpected error during upload: {str(e)}",
                details={"local_path": local_path}
            )
    
    def file_exists(self, s3_url: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            s3_url: S3 URL
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket_name, object_key = S3URLParser.parse(s3_url)
            return self._file_exists(bucket_name, object_key)
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def get_file_size(self, s3_url: str) -> int:
        """
        Get file size in bytes
        
        Args:
            s3_url: S3 URL
            
        Returns:
            File size in bytes
            
        Raises:
            S3DownloadError: If file not found
        """
        try:
            bucket_name, object_key = S3URLParser.parse(s3_url)
            return self._get_file_size(bucket_name, object_key)
        except ClientError as e:
            raise S3DownloadError(
                f"Failed to get file size: {e}",
                s3_url=s3_url
            )
    
    def delete_file(self, s3_url: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_url: S3 URL
            
        Returns:
            True if deleted successfully
        """
        try:
            bucket_name, object_key = S3URLParser.parse(s3_url)
            
            logger.info(f"🗑️ Deleting from S3: {object_key}")
            
            self.s3_client.delete_object(
                Bucket=bucket_name,
                Key=object_key
            )
            
            logger.info(f"✅ Deleted: {object_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete file: {e}")
            return False
    
    def _file_exists(self, bucket_name: str, object_key: str) -> bool:
        """Check if file exists in S3 (internal)"""
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def _get_file_size(self, bucket_name: str, object_key: str) -> int:
        """Get file size in bytes (internal)"""
        try:
            response = self.s3_client.head_object(
                Bucket=bucket_name,
                Key=object_key
            )
            return response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise S3DownloadError(
                    f"File not found: {object_key}",
                    details={
                        "bucket": bucket_name,
                        "key": object_key
                    }
                )
            raise
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


class ProgressCallback:
    """Callback for tracking download/upload progress"""
    
    def __init__(self, total_size: int, filename: str):
        self.total_size = total_size
        self.filename = filename
        self.transferred = 0
        self.last_percentage = 0
    
    def __call__(self, bytes_transferred: int):
        """Called by boto3 during transfer"""
        self.transferred += bytes_transferred
        
        if self.total_size > 0:
            percentage = (self.transferred / self.total_size) * 100
            
            # Log progress every 10%
            if percentage - self.last_percentage >= 10:
                logger.info(
                    f"   Progress: {percentage:.0f}% "
                    f"({self._format_size(self.transferred)} / "
                    f"{self._format_size(self.total_size)})"
                )
                self.last_percentage = percentage
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


# Singleton instance
_s3_service_instance = None


def get_s3_service() -> S3Service:
    """
    Get S3Service singleton instance
    
    Returns:
        S3Service instance
    """
    global _s3_service_instance
    if _s3_service_instance is None:
        _s3_service_instance = S3Service()
    return _s3_service_instance
