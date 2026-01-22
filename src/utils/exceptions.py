"""
Custom exceptions for ASR Worker

Centralized error handling for better debugging and error tracking.
"""


class ASRWorkerError(Exception):
    """Base exception for all ASR Worker errors"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class SQSMessageError(ASRWorkerError):
    """Exception raised for invalid SQS message format or content"""
    
    def __init__(self, message: str, message_body: str = None, details: dict = None):
        super().__init__(message, details)
        self.message_body = message_body


class S3DownloadError(ASRWorkerError):
    """Exception raised when S3 file download fails"""
    
    def __init__(self, message: str, s3_url: str = None, details: dict = None):
        super().__init__(message, details)
        self.s3_url = s3_url


class S3UploadError(ASRWorkerError):
    """Exception raised when S3 file upload fails"""
    
    def __init__(self, message: str, s3_url: str = None, details: dict = None):
        super().__init__(message, details)
        self.s3_url = s3_url


class AudioProcessingError(ASRWorkerError):
    """Exception raised during audio processing/normalization"""
    
    def __init__(self, message: str, audio_path: str = None, details: dict = None):
        super().__init__(message, details)
        self.audio_path = audio_path


class AudioValidationError(ASRWorkerError):
    """Exception raised when audio file validation fails"""
    
    def __init__(self, message: str, audio_path: str = None, validation_errors: list = None):
        details = {"validation_errors": validation_errors} if validation_errors else None
        super().__init__(message, details)
        self.audio_path = audio_path
        self.validation_errors = validation_errors or []


class ASRProcessingError(ASRWorkerError):
    """Exception raised during ASR/transcription processing"""
    
    def __init__(self, message: str, audio_path: str = None, engine: str = None, details: dict = None):
        super().__init__(message, details)
        self.audio_path = audio_path
        self.engine = engine


class DiarizationError(ASRWorkerError):
    """Exception raised during speaker diarization"""
    
    def __init__(self, message: str, audio_path: str = None, model: str = None, details: dict = None):
        super().__init__(message, details)
        self.audio_path = audio_path
        self.model = model


class WebhookError(ASRWorkerError):
    """Exception raised when webhook callback fails"""
    
    def __init__(self, message: str, webhook_url: str = None, status_code: int = None, details: dict = None):
        super().__init__(message, details)
        self.webhook_url = webhook_url
        self.status_code = status_code


class ConfigurationError(ASRWorkerError):
    """Exception raised for configuration/settings errors"""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        super().__init__(message, details)
        self.config_key = config_key


class TimeoutError(ASRWorkerError):
    """Exception raised when operation times out"""
    
    def __init__(self, message: str, operation: str = None, timeout_seconds: int = None, details: dict = None):
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ModelLoadError(ASRWorkerError):
    """Exception raised when AI model loading fails"""
    
    def __init__(self, message: str, model_name: str = None, details: dict = None):
        super().__init__(message, details)
        self.model_name = model_name


class RetryExhaustedError(ASRWorkerError):
    """Exception raised when retry attempts are exhausted"""
    
    def __init__(self, message: str, operation: str = None, retry_count: int = None, details: dict = None):
        super().__init__(message, details)
        self.operation = operation
        self.retry_count = retry_count


class ValidationError(ASRWorkerError):
    """Exception raised for general validation errors"""
    
    def __init__(self, message: str, field: str = None, value: any = None, details: dict = None):
        super().__init__(message, details)
        self.field = field
        self.value = value
