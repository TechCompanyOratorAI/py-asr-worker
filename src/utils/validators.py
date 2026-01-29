"""
Input validation functions for ASR Worker

Validates SQS messages, configuration, and other inputs.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

from .exceptions import ValidationError, SQSMessageError


class SQSMessageValidator:
    """Validator for SQS messages from Node API"""
    
    REQUIRED_FIELDS = [
        'presentationId',
        'jobId',
        'audioUrl',
        'queueType',
        'sentAt',
        'version'
    ]
    
    VALID_QUEUE_TYPES = ['asr']
    
    S3_URL_PATTERN = re.compile(
        r'^s3://[a-z0-9][a-z0-9\-]*[a-z0-9]/.*$',
        re.IGNORECASE
    )
    
    @classmethod
    def validate(cls, message_body: str) -> Dict[str, Any]:
        """
        Validate SQS message format and content
        
        Args:
            message_body: Raw SQS message body (JSON string)
            
        Returns:
            Parsed and validated message data
            
        Raises:
            SQSMessageError: If validation fails
        """
        # Parse JSON
        try:
            data = json.loads(message_body)
        except json.JSONDecodeError as e:
            raise SQSMessageError(
                "Invalid JSON format in SQS message",
                message_body=message_body,
                details={"error": str(e)}
            )
        
        # Check required fields
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise SQSMessageError(
                f"Missing required fields in SQS message: {', '.join(missing_fields)}",
                message_body=message_body,
                details={"missing_fields": missing_fields, "received_data": data}
            )
        
        # Validate queue type
        if data['queueType'] not in cls.VALID_QUEUE_TYPES:
            raise SQSMessageError(
                f"Invalid queue type: {data['queueType']}",
                message_body=message_body,
                details={
                    "received": data['queueType'],
                    "expected": cls.VALID_QUEUE_TYPES
                }
            )
        
        # Validate IDs are positive integers
        cls._validate_positive_int(data, 'presentationId', message_body)
        cls._validate_positive_int(data, 'jobId', message_body)
        
        # Validate S3 URL
        cls._validate_s3_url(data['audioUrl'], message_body)
        
        # Validate version format (semver-like)
        cls._validate_version(data['version'], message_body)
        
        return data
    
    @classmethod
    def _validate_positive_int(cls, data: dict, field: str, message_body: str) -> None:
        """Validate field is a positive integer"""
        value = data.get(field)
        
        if not isinstance(value, int):
            raise SQSMessageError(
                f"Field '{field}' must be an integer",
                message_body=message_body,
                details={"field": field, "value": value, "type": type(value).__name__}
            )
        
        if value <= 0:
            raise SQSMessageError(
                f"Field '{field}' must be a positive integer",
                message_body=message_body,
                details={"field": field, "value": value}
            )
    
    @classmethod
    def _validate_s3_url(cls, url: str, message_body: str) -> None:
        """Validate S3 URL format (accepts both s3:// and https:// formats)"""
        if not isinstance(url, str):
            raise SQSMessageError(
                "Audio URL must be a string",
                message_body=message_body,
                details={"audioUrl": url, "type": type(url).__name__}
            )
        
        if not url or not url.strip():
            raise SQSMessageError(
                "Audio URL cannot be empty",
                message_body=message_body,
                details={"audioUrl": url}
            )
        
        # Accept both s3:// and https:// S3 URLs
        # s3://bucket/key or https://bucket.s3.region.amazonaws.com/key
        if not (cls.S3_URL_PATTERN.match(url) or url.startswith('https://')):
            raise SQSMessageError(
                "Invalid S3 URL format",
                message_body=message_body,
                details={
                    "audioUrl": url,
                    "expected_format": "s3://bucket/key or https://bucket.s3.region.amazonaws.com/key"
                }
            )
    
    @classmethod
    def _validate_version(cls, version: str, message_body: str) -> None:
        """Validate version format"""
        if not isinstance(version, str):
            raise SQSMessageError(
                "Version must be a string",
                message_body=message_body,
                details={"version": version, "type": type(version).__name__}
            )
        
        # Accept simple version format like "1.0"
        if not re.match(r'^\d+\.\d+(\.\d+)?$', version):
            raise SQSMessageError(
                "Invalid version format",
                message_body=message_body,
                details={
                    "version": version,
                    "expected_format": "X.Y or X.Y.Z"
                }
            )


class AudioValidator:
    """Validator for audio files"""
    
    VALID_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.mp4', '.avi', '.mkv', '.mov']
    
    MAX_FILE_SIZE_MB = 500  # 500 MB
    MIN_FILE_SIZE_KB = 10   # 10 KB
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> None:
        """
        Validate audio file path
        
        Args:
            file_path: Path to audio file
            
        Raises:
            ValidationError: If validation fails
        """
        if not file_path:
            raise ValidationError(
                "Audio file path cannot be empty",
                field="file_path",
                value=file_path
            )
        
        # Check extension
        extension = cls._get_extension(file_path)
        if extension.lower() not in cls.VALID_EXTENSIONS:
            raise ValidationError(
                f"Invalid audio file extension: {extension}",
                field="file_path",
                value=file_path,
                details={
                    "extension": extension,
                    "valid_extensions": cls.VALID_EXTENSIONS
                }
            )
    
    @classmethod
    def validate_file_size(cls, file_size_bytes: int, file_path: str = None) -> None:
        """
        Validate audio file size
        
        Args:
            file_size_bytes: File size in bytes
            file_path: Optional file path for error message
            
        Raises:
            ValidationError: If file size is invalid
        """
        max_bytes = cls.MAX_FILE_SIZE_MB * 1024 * 1024
        min_bytes = cls.MIN_FILE_SIZE_KB * 1024
        
        if file_size_bytes > max_bytes:
            raise ValidationError(
                f"Audio file too large: {file_size_bytes / (1024*1024):.2f} MB",
                field="file_size",
                value=file_size_bytes,
                details={
                    "file_path": file_path,
                    "max_size_mb": cls.MAX_FILE_SIZE_MB,
                    "actual_size_mb": file_size_bytes / (1024*1024)
                }
            )
        
        if file_size_bytes < min_bytes:
            raise ValidationError(
                f"Audio file too small: {file_size_bytes / 1024:.2f} KB",
                field="file_size",
                value=file_size_bytes,
                details={
                    "file_path": file_path,
                    "min_size_kb": cls.MIN_FILE_SIZE_KB,
                    "actual_size_kb": file_size_bytes / 1024
                }
            )
    
    @classmethod
    def validate_duration(cls, duration_seconds: float, max_duration: int) -> None:
        """
        Validate audio duration
        
        Args:
            duration_seconds: Audio duration in seconds
            max_duration: Maximum allowed duration in seconds
            
        Raises:
            ValidationError: If duration is invalid
        """
        if duration_seconds <= 0:
            raise ValidationError(
                "Audio duration must be positive",
                field="duration",
                value=duration_seconds
            )
        
        if duration_seconds > max_duration:
            raise ValidationError(
                f"Audio duration too long: {duration_seconds / 60:.1f} minutes",
                field="duration",
                value=duration_seconds,
                details={
                    "max_duration_minutes": max_duration / 60,
                    "actual_duration_minutes": duration_seconds / 60
                }
            )
    
    @staticmethod
    def _get_extension(file_path: str) -> str:
        """Extract file extension from path"""
        import os
        _, ext = os.path.splitext(file_path)
        return ext


class S3URLParser:
    """Parser and validator for S3 URLs"""
    
    @classmethod
    def parse(cls, s3_url: str) -> Tuple[str, str]:
        """
        Parse S3 URL into bucket and key (accepts both s3:// and https:// formats)
        
        Args:
            s3_url: S3 URL in format s3://bucket/key or https://bucket.s3.region.amazonaws.com/key
            
        Returns:
            Tuple of (bucket_name, object_key)
            
        Raises:
            ValidationError: If URL format is invalid
        """
        # Handle HTTPS S3 URLs: https://bucket.s3.region.amazonaws.com/key
        if s3_url.startswith('https://'):
            from urllib.parse import urlparse
            parsed = urlparse(s3_url)
            
            # Extract bucket from hostname (bucket.s3.region.amazonaws.com)
            hostname_parts = parsed.hostname.split('.')
            if len(hostname_parts) >= 4 and hostname_parts[1] == 's3':
                bucket_name = hostname_parts[0]
                object_key = parsed.path.lstrip('/')
                
                if not bucket_name or not object_key:
                    raise ValidationError(
                        "Invalid S3 HTTPS URL format",
                        field="s3_url",
                        value=s3_url,
                        details={"expected_format": "https://bucket.s3.region.amazonaws.com/path/to/file"}
                    )
                
                return bucket_name, object_key
            else:
                raise ValidationError(
                    "Invalid S3 HTTPS URL format",
                    field="s3_url",
                    value=s3_url,
                    details={"expected_format": "https://bucket.s3.region.amazonaws.com/path/to/file"}
                )
        
        # Handle s3:// format
        if not s3_url.startswith('s3://'):
            raise ValidationError(
                "S3 URL must start with 's3://' or 'https://'",
                field="s3_url",
                value=s3_url
            )
        
        # Remove s3:// prefix
        path = s3_url[5:]
        
        # Split into bucket and key
        parts = path.split('/', 1)
        
        if len(parts) < 2:
            raise ValidationError(
                "S3 URL must include bucket and key",
                field="s3_url",
                value=s3_url,
                details={"expected_format": "s3://bucket/path/to/file"}
            )
        
        bucket_name = parts[0]
        object_key = parts[1]
        
        # Validate bucket name
        if not bucket_name:
            raise ValidationError(
                "S3 bucket name cannot be empty",
                field="s3_url",
                value=s3_url
            )
        
        # Validate object key
        if not object_key:
            raise ValidationError(
                "S3 object key cannot be empty",
                field="s3_url",
                value=s3_url
            )
        
        return bucket_name, object_key
    
    @classmethod
    def validate_bucket_name(cls, bucket_name: str) -> None:
        """
        Validate S3 bucket name according to AWS rules
        
        Args:
            bucket_name: S3 bucket name
            
        Raises:
            ValidationError: If bucket name is invalid
        """
        # AWS S3 bucket naming rules
        if len(bucket_name) < 3 or len(bucket_name) > 63:
            raise ValidationError(
                "S3 bucket name must be between 3 and 63 characters",
                field="bucket_name",
                value=bucket_name
            )
        
        if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', bucket_name):
            raise ValidationError(
                "Invalid S3 bucket name format",
                field="bucket_name",
                value=bucket_name,
                details={
                    "rules": [
                        "Must start and end with lowercase letter or number",
                        "Can contain lowercase letters, numbers, and hyphens",
                        "Cannot contain consecutive periods"
                    ]
                }
            )


class ConfigValidator:
    """Validator for configuration settings"""
    
    @classmethod
    def validate_required_env_vars(cls, required_vars: List[str], settings: dict) -> None:
        """
        Validate required environment variables are set
        
        Args:
            required_vars: List of required variable names
            settings: Settings dictionary
            
        Raises:
            ValidationError: If any required variable is missing
        """
        missing_vars = []
        
        for var in required_vars:
            value = settings.get(var)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValidationError(
                f"Missing required environment variables: {', '.join(missing_vars)}",
                field="environment",
                value=missing_vars,
                details={
                    "missing_vars": missing_vars,
                    "hint": "Check your .env file"
                }
            )
    
    @classmethod
    def validate_positive_integer(cls, value: Any, field_name: str, min_value: int = 1) -> None:
        """
        Validate value is a positive integer
        
        Args:
            value: Value to validate
            field_name: Name of field for error message
            min_value: Minimum allowed value
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            raise ValidationError(
                f"{field_name} must be an integer",
                field=field_name,
                value=value,
                details={"type": type(value).__name__}
            )
        
        if value < min_value:
            raise ValidationError(
                f"{field_name} must be at least {min_value}",
                field=field_name,
                value=value,
                details={"min_value": min_value}
            )
