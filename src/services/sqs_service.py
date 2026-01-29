"""
SQS Service - AWS SQS message queue operations

Handles polling messages from SQS, parsing, and deletion.
"""

import json
import boto3
from typing import List, Dict, Optional, Any
from botocore.exceptions import ClientError, NoCredentialsError

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import SQSMessageError, ConfigurationError
from utils.validators import SQSMessageValidator

logger = get_logger(__name__)


class SQSMessage:
    """Wrapper for SQS message with parsed data"""
    
    def __init__(self, raw_message: Dict, parsed_data: Dict):
        """
        Initialize SQS message wrapper
        
        Args:
            raw_message: Raw SQS message from boto3
            parsed_data: Parsed and validated message data
        """
        self.raw_message = raw_message
        self.data = parsed_data
        
        # SQS metadata
        self.message_id = raw_message.get('MessageId')
        self.receipt_handle = raw_message.get('ReceiptHandle')
        self.message_attributes = raw_message.get('MessageAttributes', {})
        
        # Job data
        self.job_id = parsed_data.get('jobId')
        self.presentation_id = parsed_data.get('presentationId')
        self.audio_url = parsed_data.get('audioUrl')
        self.metadata = parsed_data.get('metadata', {})
    
    def __repr__(self):
        return f"<SQSMessage job_id={self.job_id} presentation_id={self.presentation_id}>"


class SQSService:
    """Service for AWS SQS operations"""
    
    def __init__(self):
        """Initialize SQS client"""
        self.queue_url = settings.AWS_SQS_ASR_QUEUE_URL
        
        if not self.queue_url:
            raise ConfigurationError(
                "AWS_SQS_ASR_QUEUE_URL not configured",
                config_key="AWS_SQS_ASR_QUEUE_URL",
                details={"hint": "Set AWS_SQS_ASR_QUEUE_URL in .env"}
            )
        
        try:
            self.sqs_client = boto3.client(
                'sqs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            logger.info("✅ SQS client initialized")
            logger.info(f"   - Queue URL: {self.queue_url}")
            logger.info(f"   - Region: {settings.AWS_REGION}")
            logger.info(f"   - Max messages: {settings.MAX_MESSAGES}")
            logger.info(f"   - Wait time: {settings.WAIT_TIME_SECONDS}s")
            logger.info(f"   - Visibility timeout: {settings.VISIBILITY_TIMEOUT}s")
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            raise ConfigurationError(
                "AWS credentials not configured",
                details={
                    "hint": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
                }
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQS client: {e}")
            raise ConfigurationError(f"Failed to initialize SQS client: {e}")
    
    def poll_messages(
        self,
        max_messages: Optional[int] = None,
        wait_time_seconds: Optional[int] = None
    ) -> List[SQSMessage]:
        """
        Poll messages from SQS queue using long polling
        
        Args:
            max_messages: Maximum number of messages to receive (default: settings.MAX_MESSAGES)
            wait_time_seconds: Long polling wait time (default: settings.WAIT_TIME_SECONDS)
            
        Returns:
            List of SQSMessage objects
            
        Raises:
            SQSMessageError: If polling fails
        """
        try:
            if max_messages is None:
                max_messages = settings.MAX_MESSAGES
            
            if wait_time_seconds is None:
                wait_time_seconds = settings.WAIT_TIME_SECONDS
            
            # Log polling attempt (debug level to avoid spam)
            logger.debug(f"📥 Polling SQS queue...")
            logger.debug(f"   - Max messages: {max_messages}")
            logger.debug(f"   - Wait time: {wait_time_seconds}s")
            
            # Receive messages from SQS
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=settings.VISIBILITY_TIMEOUT,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
            
            # Extract messages
            raw_messages = response.get('Messages', [])
            
            if not raw_messages:
                logger.debug("   - No messages available")
                return []
            
            logger.info(f"✅ Received {len(raw_messages)} message(s) from SQS")
            
            # Parse and validate messages
            sqs_messages = []
            
            for raw_msg in raw_messages:
                try:
                    # Parse message
                    sqs_message = self.parse_message(raw_msg)
                    sqs_messages.append(sqs_message)
                    
                    logger.info(f"   - Message: job_id={sqs_message.job_id}, "
                               f"presentation_id={sqs_message.presentation_id}")
                    
                except SQSMessageError as e:
                    logger.error(f"❌ Invalid message: {e}")
                    logger.error(f"   - Message ID: {raw_msg.get('MessageId')}")
                    
                    # Delete invalid message to avoid reprocessing
                    try:
                        self.delete_message(raw_msg)
                        logger.info("   - Invalid message deleted from queue")
                    except Exception as del_error:
                        logger.error(f"   - Failed to delete invalid message: {del_error}")
                    
                    continue
            
            return sqs_messages
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"❌ SQS polling failed: {error_code} - {error_message}")
            raise SQSMessageError(
                f"SQS polling failed: {error_message}",
                details={
                    "error_code": error_code,
                    "queue_url": self.queue_url
                }
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error polling SQS: {e}", exc_info=True)
            raise SQSMessageError(f"Unexpected error polling SQS: {str(e)}")
    
    def parse_message(self, raw_message: Dict) -> SQSMessage:
        """
        Parse and validate SQS message
        
        Args:
            raw_message: Raw SQS message from boto3
            
        Returns:
            SQSMessage object
            
        Raises:
            SQSMessageError: If message format is invalid
        """
        try:
            # Extract message body
            message_body = raw_message.get('Body')
            
            if not message_body:
                raise SQSMessageError(
                    "Empty message body",
                    message_body="<empty>"
                )
            
            # Validate and parse using validator
            parsed_data = SQSMessageValidator.validate(message_body)
            
            # Create SQSMessage wrapper
            sqs_message = SQSMessage(raw_message, parsed_data)
            
            return sqs_message
            
        except SQSMessageError:
            raise
        except Exception as e:
            raise SQSMessageError(
                f"Failed to parse message: {str(e)}",
                message_body=raw_message.get('Body', '<empty>')
            )
    
    def delete_message(self, message: Any) -> bool:
        """
        Delete message from SQS queue
        
        Args:
            message: SQSMessage object or raw message dict
            
        Returns:
            True if deleted successfully
            
        Raises:
            SQSMessageError: If deletion fails
        """
        try:
            # Extract receipt handle
            if isinstance(message, SQSMessage):
                receipt_handle = message.receipt_handle
                message_id = message.message_id
            else:
                # Raw message dict
                receipt_handle = message.get('ReceiptHandle')
                message_id = message.get('MessageId')
            
            if not receipt_handle:
                raise SQSMessageError("Missing receipt handle for message deletion")
            
            logger.info(f"🗑️ Deleting message from SQS...")
            logger.info(f"   - Message ID: {message_id}")
            
            # Delete from queue
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info(f"✅ Message deleted successfully")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"❌ Failed to delete message: {error_code} - {error_message}")
            raise SQSMessageError(
                f"Failed to delete message: {error_message}",
                details={"error_code": error_code}
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting message: {e}", exc_info=True)
            raise SQSMessageError(f"Unexpected error deleting message: {str(e)}")
    
    def change_visibility(
        self,
        message: Any,
        visibility_timeout: int
    ) -> bool:
        """
        Change visibility timeout for a message
        
        Useful for extending processing time for long-running jobs.
        
        Args:
            message: SQSMessage object or raw message dict
            visibility_timeout: New visibility timeout in seconds
            
        Returns:
            True if successful
            
        Raises:
            SQSMessageError: If operation fails
        """
        try:
            # Extract receipt handle
            if isinstance(message, SQSMessage):
                receipt_handle = message.receipt_handle
                message_id = message.message_id
            else:
                receipt_handle = message.get('ReceiptHandle')
                message_id = message.get('MessageId')
            
            if not receipt_handle:
                raise SQSMessageError("Missing receipt handle")
            
            logger.info(f"⏱️ Changing message visibility...")
            logger.info(f"   - Message ID: {message_id}")
            logger.info(f"   - New timeout: {visibility_timeout}s")
            
            # Change visibility
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            
            logger.info(f"✅ Visibility timeout updated")
            return True
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"❌ Failed to change visibility: {error_message}")
            raise SQSMessageError(f"Failed to change visibility: {error_message}")
        except Exception as e:
            logger.error(f"❌ Unexpected error changing visibility: {e}")
            raise SQSMessageError(f"Unexpected error: {str(e)}")
    
    def get_queue_attributes(self) -> Dict[str, Any]:
        """
        Get queue attributes (message count, etc.)
        
        Returns:
            Dictionary of queue attributes
        """
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            attributes = response.get('Attributes', {})
            
            return {
                'approximate_messages': int(attributes.get('ApproximateNumberOfMessages', 0)),
                'approximate_messages_not_visible': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'approximate_messages_delayed': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                'queue_arn': attributes.get('QueueArn'),
                'created_timestamp': attributes.get('CreatedTimestamp'),
                'visibility_timeout': int(attributes.get('VisibilityTimeout', 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue attributes: {e}")
            return {}
    
    def purge_queue(self) -> bool:
        """
        Purge all messages from queue (USE WITH CAUTION!)
        
        Returns:
            True if successful
        """
        try:
            logger.warning("⚠️ PURGING QUEUE - All messages will be deleted!")
            
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            
            logger.warning("✅ Queue purged")
            return True
            
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"❌ Failed to purge queue: {error_message}")
            return False


class MockSQSService:
    """Mock SQS service for testing"""
    
    def __init__(self):
        logger.warning("⚠️ Using MOCK SQS Service (for testing only)")
        self._mock_messages = []
        self._message_counter = 0
    
    def poll_messages(self, **kwargs) -> List[SQSMessage]:
        """Return mock messages"""
        logger.debug("📥 Mock: Polling SQS queue...")
        
        # Return empty most of the time
        if not self._mock_messages and self._message_counter < 1:
            # Generate one mock message
            mock_msg = self._create_mock_message()
            self._message_counter += 1
            logger.info(f"✅ Mock: Generated 1 message")
            return [mock_msg]
        
        logger.debug("   - Mock: No messages")
        return []
    
    def parse_message(self, raw_message: Dict) -> SQSMessage:
        """Parse mock message"""
        message_body = raw_message.get('Body', '{}')
        parsed_data = json.loads(message_body)
        return SQSMessage(raw_message, parsed_data)
    
    def delete_message(self, message: Any) -> bool:
        """Mock delete"""
        message_id = message.message_id if isinstance(message, SQSMessage) else message.get('MessageId')
        logger.info(f"🗑️ Mock: Deleting message {message_id}")
        logger.info("✅ Mock: Message deleted")
        return True
    
    def change_visibility(self, message: Any, visibility_timeout: int) -> bool:
        """Mock visibility change"""
        logger.info(f"⏱️ Mock: Changing visibility to {visibility_timeout}s")
        return True
    
    def get_queue_attributes(self) -> Dict:
        """Mock queue attributes"""
        return {
            'approximate_messages': 0,
            'approximate_messages_not_visible': 0,
            'approximate_messages_delayed': 0,
            'queue_arn': 'arn:aws:sqs:mock:123456789012:mock-queue',
            'visibility_timeout': 300
        }
    
    def purge_queue(self) -> bool:
        """Mock purge"""
        logger.warning("⚠️ Mock: Queue purge requested (no-op)")
        return True
    
    def _create_mock_message(self) -> SQSMessage:
        """Create a mock SQS message"""
        message_body = json.dumps({
            "presentationId": 123,
            "jobId": 456,
            "audioUrl": "s3://mock-bucket/audio/presentation_123.mp3",
            "metadata": {
                "userId": 789,
                "courseId": 10
            },
            "queueType": "asr",
            "sentAt": "2026-01-24T10:30:00Z",
            "version": "1.0"
        })
        
        raw_message = {
            "MessageId": f"mock-message-{self._message_counter}",
            "ReceiptHandle": f"mock-receipt-{self._message_counter}",
            "Body": message_body,
            "MessageAttributes": {
                "presentationId": {
                    "DataType": "Number",
                    "StringValue": "123"
                },
                "jobId": {
                    "DataType": "Number",
                    "StringValue": "456"
                }
            }
        }
        
        parsed_data = json.loads(message_body)
        return SQSMessage(raw_message, parsed_data)


# Singleton instance
_sqs_service_instance = None


def get_sqs_service(use_mock: bool = None) -> SQSService:
    """
    Get SQSService singleton instance
    
    Args:
        use_mock: Use mock service (default: settings.MOCK_ASR)
        
    Returns:
        SQSService or MockSQSService instance
    """
    global _sqs_service_instance
    
    if _sqs_service_instance is None:
        if use_mock is None:
            use_mock = settings.MOCK_ASR
        
        if use_mock:
            _sqs_service_instance = MockSQSService()
        else:
            _sqs_service_instance = SQSService()
    
    return _sqs_service_instance
