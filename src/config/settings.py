"""
Configuration settings loaded from environment variables
"""

import os
import uuid
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # ========================================
    # PyTorch Configuration
    # ========================================
    TORCH_FORCE_WEIGHTS_ONLY_LOAD: str = Field(default="0", description="Disable PyTorch 2.6+ weights_only check")
    
    # ========================================
    # AWS Configuration
    # ========================================
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS secret access key")
    AWS_REGION: str = Field(default="ap-southeast-1", description="AWS region")
    AWS_S3_BUCKET: str = Field(..., description="S3 bucket name")
    AWS_SQS_ASR_QUEUE_URL: str = Field(..., description="SQS ASR queue URL")
    
    # ========================================
    # Node API Configuration
    # ========================================
    NODE_API_URL: str = Field(default="http://localhost:8080", description="Node API base URL")
    WEBHOOK_SECRET: str = Field(..., description="Webhook authentication secret")
    WEBHOOK_ENDPOINT: str = Field(default="/api/v1/webhooks/asr-complete", description="Webhook endpoint")
    
    # ========================================
    # ASR Configuration
    # ========================================
    ASR_ENGINE: str = Field(default="whisper", description="ASR engine: whisper, google, azure")
    
    # Whisper Configuration
    WHISPER_MODEL: str = Field(default="base", description="Whisper model: tiny, base, small, medium, large")
    WHISPER_LANGUAGE: str = Field(default="vi", description="Language code")
    WHISPER_DEVICE: str = Field(default="cpu", description="Device: cpu or cuda")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="Compute type: int8, float16, float32")
    
    # ========================================
    # Speaker Diarization Configuration
    # ========================================
    DIARIZATION_ENABLED: bool = Field(default=True, description="Enable speaker diarization")
    DIARIZATION_MODEL: str = Field(default="pyannote/speaker-diarization-3.1", description="Diarization model")
    MIN_SPEAKERS: int = Field(default=1, description="Minimum number of speakers")
    MAX_SPEAKERS: int = Field(default=5, description="Maximum number of speakers")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, description="Hugging Face API token")
    
    # ========================================
    # Worker Configuration
    # ========================================
    POLL_INTERVAL: int = Field(default=5, description="Seconds between SQS polls")
    MAX_MESSAGES: int = Field(default=1, description="Max messages per poll")
    VISIBILITY_TIMEOUT: int = Field(default=300, description="SQS visibility timeout (seconds)")
    WAIT_TIME_SECONDS: int = Field(default=20, description="SQS long polling wait time")
    
    MAX_WORKERS: int = Field(default=1, description="Number of concurrent workers")
    MAX_RETRIES: int = Field(default=3, description="Max retry attempts")
    RETRY_DELAY: int = Field(default=5, description="Seconds between retries")
    
    # Timeouts
    DOWNLOAD_TIMEOUT: int = Field(default=300, description="Download timeout (seconds)")
    PROCESSING_TIMEOUT: int = Field(default=600, description="Processing timeout (seconds)")
    WEBHOOK_TIMEOUT: int = Field(default=30, description="Webhook timeout (seconds)")
    
    # ========================================
    # Logging Configuration
    # ========================================
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(default="text", description="Log format: json or text")
    LOG_FILE: str = Field(default="logs/asr-worker.log", description="Log file path")
    LOG_MAX_BYTES: int = Field(default=10485760, description="Max log file size (bytes)")
    LOG_BACKUP_COUNT: int = Field(default=5, description="Number of log backup files")
    
    # ========================================
    # Performance Tuning
    # ========================================
    CHUNK_SIZE: int = Field(default=8192, description="File download chunk size (bytes)")
    TEMP_DIR: str = Field(default="temp", description="Temporary files directory")
    CLEANUP_TEMP_FILES: bool = Field(default=True, description="Clean up temp files after processing")
    
    MAX_AUDIO_LENGTH: int = Field(default=3600, description="Max audio length (seconds)")
    AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate (Hz)")
    
    # ========================================
    # Development/Debug
    # ========================================
    DEBUG: bool = Field(default=False, description="Debug mode")
    DRY_RUN: bool = Field(default=False, description="Dry run mode (no webhook/delete)")
    MOCK_ASR: bool = Field(default=False, description="Use mock ASR data")
    
    # Worker ID (auto-generated)
    WORKER_ID: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @property
    def webhook_url(self) -> str:
        """Get full webhook URL"""
        return f"{self.NODE_API_URL}{self.WEBHOOK_ENDPOINT}"


# Global settings instance
settings = Settings()
