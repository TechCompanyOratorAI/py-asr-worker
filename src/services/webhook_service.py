"""
Webhook Service - Send processing results to Node API

Handles webhook callbacks with authentication and retry logic.
"""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import WebhookError, RetryExhaustedError

logger = get_logger(__name__)


class WebhookService:
    """Service for sending webhook callbacks to Node API"""
    
    def __init__(self):
        """Initialize webhook service"""
        self.node_api_url = settings.NODE_API_URL
        self.webhook_secret = settings.WEBHOOK_SECRET
        self.webhook_endpoint = settings.WEBHOOK_ENDPOINT
        self.timeout = settings.WEBHOOK_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        
        # Build full webhook URL
        self.webhook_url = f"{self.node_api_url}{self.webhook_endpoint}"
        
        logger.info("✅ Webhook service initialized")
        logger.info(f"   - URL: {self.webhook_url}")
        logger.info(f"   - Timeout: {self.timeout}s")
        logger.info(f"   - Max retries: {self.max_retries}")
    
    def send_asr_complete(
        self,
        job_id: int,
        presentation_id: int,
        transcript_segments: List[Dict],
        speakers: List[Dict],
        metadata: Dict[str, Any],
        status: str = "completed"
    ) -> bool:
        """
        Send ASR completion webhook to Node API
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            transcript_segments: List of transcript segments with speakers
            speakers: List of speaker info
            metadata: Additional metadata (duration, model, etc.)
            status: Job status ("completed" or "failed")
            
        Returns:
            True if webhook sent successfully
            
        Raises:
            WebhookError: If webhook fails after all retries
        """
        # Build fullText from segments
        full_text = "\n\n".join([
            f"[{seg.get('speakerLabel', 'UNKNOWN')}] {seg.get('text', '')}"
            for seg in transcript_segments
        ])
        
        # Convert segments to backend format
        formatted_segments = [
            {
                "order": idx + 1,
                "startTimestamp": seg.get("start", 0),
                "endTimestamp": seg.get("end", 0),
                "text": seg.get("text", ""),
                "confidence": seg.get("confidence", 0.0)
            }
            for idx, seg in enumerate(transcript_segments)
        ]
        
        # Build speaker mappings
        segment_speaker_mappings = [
            {
                "order": idx + 1,
                "aiSpeakerLabel": seg.get("speakerLabel", "UNKNOWN")
            }
            for idx, seg in enumerate(transcript_segments)
        ]
        
        # Format speakers for diarization
        formatted_speakers = [
            {
                "aiSpeakerLabel": spk.get("aiSpeakerLabel", "UNKNOWN"),
                "segments": [],  # Not used by backend
                "metadata": {
                    "segmentCount": spk.get("segmentCount", 0),
                    "totalDuration": spk.get("totalDuration", 0),
                    "confidence": spk.get("confidence", 0.0),
                    "percentage": spk.get("percentage", 0.0)
                }
            }
            for spk in speakers
        ]
        
        # Build payload in backend-expected format
        payload = {
            "jobId": job_id,
            "presentationId": presentation_id,
            "status": status if status == "failed" else "success",
            "transcript": {
                "fullText": full_text,
                "language": metadata.get("whisperLanguage", "vi"),
                "segments": formatted_segments
            },
            "diarization": {
                "speakers": formatted_speakers,
                "segmentSpeakerMappings": segment_speaker_mappings
            },
            "metadata": {
                **metadata,
                "completedAt": datetime.utcnow().isoformat() + "Z",
                "workerName": f"ASRWorker-{settings.WORKER_ID}"
            }
        }
        
        logger.info(f"📤 Sending ASR complete webhook...")
        logger.info(f"   - Job ID: {job_id}")
        logger.info(f"   - Presentation ID: {presentation_id}")
        logger.info(f"   - Status: {status}")
        logger.info(f"   - Transcript segments: {len(transcript_segments)}")
        logger.info(f"   - Speakers: {len(speakers)}")
        logger.info(f"   - URL: {self.webhook_url}")
        
        try:
            # Send with retry logic
            response = self._send_with_retry(payload)
            
            logger.info(f"✅ Webhook sent successfully")
            logger.info(f"   - Status code: {response.status_code}")
            logger.info(f"   - Response: {response.text[:200]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send webhook: {e}", exc_info=True)
            raise
    
    def send_asr_failed(
        self,
        job_id: int,
        presentation_id: int,
        error_message: str,
        error_details: Optional[Dict] = None
    ) -> bool:
        """
        Send ASR failure webhook to Node API
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            error_message: Error message
            error_details: Additional error details
            
        Returns:
            True if webhook sent successfully
        """
        # Convert error_details to JSON string for API compatibility
        import json
        error_details_str = json.dumps(error_details or {}, ensure_ascii=False)
        
        payload = {
            "jobId": job_id,
            "presentationId": presentation_id,
            "status": "failed",
            "errorMessage": f"{error_message} | Details: {error_details_str}"
        }
        
        logger.warning(f"⚠️ Sending ASR failed webhook...")
        logger.warning(f"   - Job ID: {job_id}")
        logger.warning(f"   - Error: {error_message}")
        
        try:
            response = self._send_with_retry(payload)
            logger.info(f"✅ Failure webhook sent successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send failure webhook: {e}")
            # Don't raise - we already failed, just log
            return False
    
    def _send_with_retry(
        self,
        payload: Dict,
        attempt: int = 1
    ) -> requests.Response:
        """
        Send webhook with retry logic using exponential backoff
        
        Args:
            payload: Webhook payload
            attempt: Current attempt number
            
        Returns:
            Response object
            
        Raises:
            RetryExhaustedError: If all retries fail
            WebhookError: If webhook fails
        """
        try:
            # Build headers
            headers = self._build_headers()
            
            # Log attempt
            if attempt > 1:
                logger.info(f"   - Retry attempt {attempt}/{self.max_retries}")
            
            # Send POST request
            response = requests.post(
                url=self.webhook_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check response status
            if response.status_code >= 200 and response.status_code < 300:
                # Success
                return response
            
            elif response.status_code >= 400 and response.status_code < 500:
                # Client error (4xx) - don't retry
                error_msg = f"Webhook failed with client error: {response.status_code}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"   - Response: {response.text[:500]}")
                
                raise WebhookError(
                    error_msg,
                    webhook_url=self.webhook_url,
                    status_code=response.status_code,
                    details={
                        "response": response.text[:500]
                    }
                )
            
            else:
                # Server error (5xx) or other - retry
                error_msg = f"Webhook failed with status {response.status_code}"
                logger.warning(f"⚠️ {error_msg} - will retry")
                
                if attempt >= self.max_retries:
                    raise RetryExhaustedError(
                        f"Webhook failed after {self.max_retries} attempts",
                        operation="webhook",
                        retry_count=attempt,
                        details={
                            "last_status_code": response.status_code,
                            "last_response": response.text[:200]
                        }
                    )
                
                # Calculate backoff delay (exponential)
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.info(f"   - Waiting {delay}s before retry...")
                time.sleep(delay)
                
                # Retry
                return self._send_with_retry(payload, attempt + 1)
        
        except requests.exceptions.Timeout:
            error_msg = f"Webhook request timed out after {self.timeout}s"
            logger.warning(f"⚠️ {error_msg}")
            
            if attempt >= self.max_retries:
                raise RetryExhaustedError(
                    f"Webhook timed out after {self.max_retries} attempts",
                    operation="webhook",
                    retry_count=attempt
                )
            
            # Retry on timeout
            delay = self.retry_delay * (2 ** (attempt - 1))
            logger.info(f"   - Waiting {delay}s before retry...")
            time.sleep(delay)
            return self._send_with_retry(payload, attempt + 1)
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            
            if attempt >= self.max_retries:
                raise RetryExhaustedError(
                    f"Connection failed after {self.max_retries} attempts",
                    operation="webhook",
                    retry_count=attempt,
                    details={"error": str(e)}
                )
            
            # Retry on connection error
            delay = self.retry_delay * (2 ** (attempt - 1))
            logger.info(f"   - Waiting {delay}s before retry...")
            time.sleep(delay)
            return self._send_with_retry(payload, attempt + 1)
        
        except (WebhookError, RetryExhaustedError):
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            # Unexpected error
            logger.error(f"❌ Unexpected error sending webhook: {e}", exc_info=True)
            raise WebhookError(
                f"Unexpected error sending webhook: {str(e)}",
                webhook_url=self.webhook_url
            )
    
    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for webhook request
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"ASRWorker/{settings.WORKER_ID}"
        }
        
        # Add authorization header if webhook secret is configured
        if self.webhook_secret:
            headers["Authorization"] = f"Bearer {self.webhook_secret}"
        else:
            logger.warning("⚠️ WEBHOOK_SECRET not set - webhook not authenticated")
        
        return headers
    
    def test_connection(self) -> bool:
        """
        Test connection to Node API (health check)
        
        Returns:
            True if connection successful
        """
        try:
            # Try to reach the base URL (not webhook endpoint)
            health_url = f"{self.node_api_url}/health"
            
            logger.info(f"🔍 Testing connection to Node API...")
            logger.info(f"   - URL: {health_url}")
            
            response = requests.get(
                health_url,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Connection successful")
                return True
            else:
                logger.warning(f"⚠️ Unexpected status: {response.status_code}")
                return False
        
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Connection failed - Node API not reachable")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False


class MockWebhookService:
    """Mock webhook service for testing"""
    
    def __init__(self):
        logger.warning("⚠️ Using MOCK Webhook Service (for testing only)")
    
    def send_asr_complete(self, job_id, presentation_id, transcript_segments, speakers, metadata, status="completed"):
        """Mock webhook - just log"""
        logger.info(f"📤 Mock: Would send webhook for job {job_id}")
        logger.info(f"   - Transcript segments: {len(transcript_segments)}")
        logger.info(f"   - Speakers: {len(speakers)}")
        logger.info(f"   - Status: {status}")
        
        if settings.DRY_RUN:
            logger.info("   - DRY_RUN mode: Not actually sending")
        
        return True
    
    def send_asr_failed(self, job_id, presentation_id, error_message, error_details=None):
        """Mock failure webhook"""
        logger.warning(f"📤 Mock: Would send failure webhook for job {job_id}")
        logger.warning(f"   - Error: {error_message}")
        return True
    
    def test_connection(self):
        """Mock connection test"""
        logger.info("🔍 Mock: Connection test skipped")
        return True


# Singleton instance
_webhook_service_instance = None


def get_webhook_service(use_mock: bool = None) -> WebhookService:
    """
    Get WebhookService singleton instance
    
    Args:
        use_mock: Use mock service (default: settings.DRY_RUN)
        
    Returns:
        WebhookService or MockWebhookService instance
    """
    global _webhook_service_instance
    
    if _webhook_service_instance is None:
        if use_mock is None:
            use_mock = settings.DRY_RUN
        
        if use_mock:
            _webhook_service_instance = MockWebhookService()
        else:
            _webhook_service_instance = WebhookService()
    
    return _webhook_service_instance
