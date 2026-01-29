# --- DLL bootstrap (Windows + PyTorch CUDA) ---
import os, ctypes
from pathlib import Path
import torch

torch_lib = Path(torch.__file__).parent / "lib"
os.add_dll_directory(str(torch_lib))
try:
    ctypes.WinDLL(str(torch_lib / "cudnn64_9.dll"))
except Exception:
    pass  # If DLL not found, continue
# --------------------------------------------

"""
ASR Worker - Main Entry Point

This worker polls messages from AWS SQS, processes audio files,
performs speech-to-text and speaker diarization, then sends results
back to Node API via webhook.
"""

import sys
import signal
import time
from typing import Optional, Dict, Any

# Add NVIDIA cuDNN/cuBLAS to PATH and environment variables for GPU acceleration
try:
    import site
    site_packages = site.getsitepackages()[0]
    
    # Add DLL paths to PATH
    nvidia_paths = [
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "lib"),
        os.path.join(site_packages, "nvidia", "cublas", "lib"),
    ]
    
    for path in nvidia_paths:
        if os.path.exists(path) and path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    
    # Set additional environment variables for cuDNN discovery
    cudnn_path = os.path.join(site_packages, "nvidia", "cudnn")
    if os.path.exists(cudnn_path):
        os.environ["CUDNN_PATH"] = cudnn_path
        os.environ["LD_LIBRARY_PATH"] = cudnn_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        
    # Try to preload cuDNN libraries before PyTorch
    try:
        cudnn_dll = os.path.join(site_packages, "nvidia", "cudnn", "bin", "cudnn64_9.dll")
        if os.path.exists(cudnn_dll):
            ctypes.CDLL(cudnn_dll)
    except Exception:
        pass
        
except Exception:
    pass  # If adding PATH fails, continue anyway

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import (
    ASRWorkerError,
    S3DownloadError,
    AudioProcessingError,
    ASRProcessingError,
    DiarizationError,
    WebhookError
)

# Import services
from services.sqs_service import get_sqs_service, SQSMessage
from services.s3_service import get_s3_service
from services.audio_processor import get_audio_processor
from services.asr_service import get_asr_service
from services.diarization_service import get_diarization_service
from services.webhook_service import get_webhook_service

logger = get_logger(__name__)


class ASRWorker:
    """Main ASR Worker class"""
    
    def __init__(self):
        self.running = False
        self.worker_name = f"ASRWorker-{settings.WORKER_ID}"
        
        # Initialize services
        self.sqs_service = None
        self.s3_service = None
        self.audio_processor = None
        self.asr_service = None
        self.diarization_service = None
        self.webhook_service = None
        
        # Statistics
        self.jobs_processed = 0
        self.jobs_succeeded = 0
        self.jobs_failed = 0
        
    def start(self):
        """Start the worker"""
        logger.info("=" * 80)
        logger.info(f"🚀 Starting {self.worker_name}")
        logger.info("=" * 80)
        logger.info(f"📊 Configuration:")
        logger.info(f"   - ASR Engine: {settings.ASR_ENGINE}")
        logger.info(f"   - Whisper Model: {settings.WHISPER_MODEL}")
        logger.info(f"   - Whisper Device: {settings.WHISPER_DEVICE}")
        logger.info(f"   - Diarization: {settings.DIARIZATION_ENABLED}")
        logger.info(f"   - Poll Interval: {settings.POLL_INTERVAL}s")
        logger.info(f"   - Max Messages: {settings.MAX_MESSAGES}")
        logger.info(f"   - Mock Mode: {settings.MOCK_ASR}")
        logger.info(f"   - Dry Run: {settings.DRY_RUN}")
        logger.info("=" * 80)
        
        # Initialize services
        self._initialize_services()
        
        self.running = True
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("⚠️ Received keyboard interrupt")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)
    
    def _initialize_services(self):
        """Initialize all services"""
        try:
            logger.info("🔧 Initializing services...")
            
            # SQS Service
            self.sqs_service = get_sqs_service()
            logger.info("   ✅ SQS Service ready")
            
            # S3 Service
            self.s3_service = get_s3_service()
            logger.info("   ✅ S3 Service ready")
            
            # Audio Processor
            self.audio_processor = get_audio_processor()
            logger.info("   ✅ Audio Processor ready")
            
            # ASR Service (lazy load model)
            self.asr_service = get_asr_service()
            logger.info("   ✅ ASR Service ready")
            
            # Diarization Service (lazy load model)
            if settings.DIARIZATION_ENABLED:
                self.diarization_service = get_diarization_service()
                logger.info("   ✅ Diarization Service ready")
            else:
                logger.warning("   ⚠️ Diarization disabled")
            
            # Webhook Service
            self.webhook_service = get_webhook_service()
            logger.info("   ✅ Webhook Service ready")
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
            raise
    
    def _run_loop(self):
        """Main worker loop - poll and process messages"""
        logger.info("🔄 Worker started, polling for messages...")
        logger.info("")
        
        while self.running:
            try:
                # Poll SQS for messages
                messages = self.sqs_service.poll_messages(
                    max_messages=settings.MAX_MESSAGES,
                    wait_time_seconds=settings.WAIT_TIME_SECONDS
                )
                
                if not messages:
                    # No messages - continue polling
                    continue
                
                # Process each message
                for message in messages:
                    self._process_message(message)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"❌ Error in worker loop: {e}", exc_info=True)
                time.sleep(settings.POLL_INTERVAL)
    
    def _process_message(self, message: SQSMessage):
        """
        Process a single SQS message
        
        Args:
            message: SQSMessage object
        """
        job_id = message.job_id
        presentation_id = message.presentation_id
        audio_url = message.audio_url
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🎯 Processing Job {job_id}")
        logger.info("=" * 80)
        logger.info(f"   - Presentation ID: {presentation_id}")
        logger.info(f"   - Audio URL: {audio_url}")
        logger.info(f"   - Message ID: {message.message_id}")
        
        # Track processing time
        start_time = time.time()
        
        # Temporary file paths (for cleanup)
        temp_files = []
        
        try:
            # Process the job
            result = self._process_job(
                job_id=job_id,
                presentation_id=presentation_id,
                audio_url=audio_url,
                metadata=message.metadata,
                temp_files=temp_files
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result['metadata']['processingTime'] = round(processing_time, 2)
            
            # Send success webhook
            logger.info(f"📤 Sending success webhook...")
            self.webhook_service.send_asr_complete(
                job_id=job_id,
                presentation_id=presentation_id,
                transcript_segments=result['transcript'],
                speakers=result['speakers'],
                metadata=result['metadata'],
                status="completed"
            )
            
            # Delete message from queue (success)
            logger.info(f"🗑️ Deleting message from queue...")
            self.sqs_service.delete_message(message)
            
            # Update statistics
            self.jobs_processed += 1
            self.jobs_succeeded += 1
            
            logger.info("=" * 80)
            logger.info(f"✅ Job {job_id} completed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Stats: {self.jobs_succeeded} succeeded, {self.jobs_failed} failed, {self.jobs_processed} total")
            logger.info("=" * 80)
            logger.info("")
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.error("=" * 80)
            logger.error(f"❌ Job {job_id} failed after {processing_time:.2f}s")
            logger.error(f"❌ Error: {e}", exc_info=True)
            logger.error("=" * 80)
            
            # Send failure webhook
            try:
                logger.warning(f"📤 Sending failure webhook...")
                self.webhook_service.send_asr_failed(
                    job_id=job_id,
                    presentation_id=presentation_id,
                    error_message=str(e),
                    error_details={
                        'error_type': type(e).__name__,
                        'audio_url': audio_url,
                        'processing_time': round(processing_time, 2)
                    }
                )
            except Exception as webhook_error:
                logger.error(f"❌ Failed to send failure webhook: {webhook_error}")
            
            # Update statistics
            self.jobs_processed += 1
            self.jobs_failed += 1
            
            # DO NOT delete message - allow retry after visibility timeout
            logger.warning(f"⚠️ Message NOT deleted - will retry after visibility timeout")
            logger.info(f"📊 Stats: {self.jobs_succeeded} succeeded, {self.jobs_failed} failed, {self.jobs_processed} total")
            logger.info("")
            
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_files)
    
    def _process_job(
        self,
        job_id: int,
        presentation_id: int,
        audio_url: str,
        metadata: Dict[str, Any],
        temp_files: list
    ) -> Dict[str, Any]:
        """
        Process ASR job pipeline
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            audio_url: S3 URL to audio file
            metadata: Job metadata
            temp_files: List to track temporary files for cleanup
            
        Returns:
            Dictionary with transcript, speakers, and metadata
        """
        # Step 1: Download audio from S3
        logger.info(f"📥 Step 1/6: Downloading audio from S3...")
        audio_path = self.s3_service.download_file(
            s3_url=audio_url,
            local_dir=settings.TEMP_DIR
        )
        temp_files.append(audio_path)
        
        # Step 2: Validate and normalize audio
        logger.info(f"🎵 Step 2/6: Validating and normalizing audio...")
        
        # Validate audio
        is_valid, errors = self.audio_processor.validate_audio(audio_path)
        if not is_valid:
            raise AudioProcessingError(
                f"Audio validation failed: {', '.join(errors)}",
                audio_path=audio_path,
                details={'validation_errors': errors}
            )
        
        # Get audio info
        audio_info = self.audio_processor.get_audio_info(audio_path)
        logger.info(f"   - Duration: {audio_info['duration']:.2f}s")
        logger.info(f"   - Sample rate: {audio_info['sample_rate']} Hz")
        logger.info(f"   - Channels: {audio_info['channels']}")
        
        # Normalize audio for ASR
        normalized_path = self.audio_processor.normalize_audio(
            input_path=audio_path,
            output_dir=settings.TEMP_DIR,
            sample_rate=settings.AUDIO_SAMPLE_RATE,
            channels=1  # Mono
        )
        temp_files.append(normalized_path)
        
        # Step 3: ASR Transcription
        logger.info(f"🎤 Step 3/6: Running ASR transcription...")
        transcript_segments = self.asr_service.transcribe(
            audio_path=normalized_path,
            language=settings.WHISPER_LANGUAGE,
            beam_size=5,
            vad_filter=True
        )
        
        # Step 4: Speaker Diarization
        if settings.DIARIZATION_ENABLED and self.diarization_service:
            logger.info(f"👥 Step 4/6: Running speaker diarization...")
            diarization_segments = self.diarization_service.diarize(
                audio_path=normalized_path,
                min_speakers=settings.MIN_SPEAKERS,
                max_speakers=settings.MAX_SPEAKERS
            )
            
            # Step 5: Merge transcript with diarization
            logger.info(f"🔗 Step 5/6: Merging transcript with speakers...")
            merged_segments = self.diarization_service.merge_with_transcript(
                transcript_segments=transcript_segments,
                diarization_segments=diarization_segments,
                overlap_threshold=0.5
            )
            
            # Get speaker info
            speakers_info = self.diarization_service.get_speaker_info(merged_segments)
            
        else:
            # No diarization - assign UNKNOWN speaker
            logger.warning(f"⚠️ Step 4-5/6: Diarization disabled - assigning UNKNOWN speaker")
            
            from services.diarization_service import TranscriptWithSpeaker, SpeakerInfo
            
            merged_segments = [
                TranscriptWithSpeaker(
                    id=seg.id,
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    speaker_label="UNKNOWN",
                    confidence=0.0,
                    transcript_confidence=seg.confidence
                )
                for seg in transcript_segments
            ]
            
            speakers_info = [
                SpeakerInfo(
                    speaker_label="UNKNOWN",
                    segment_count=len(merged_segments),
                    total_duration=audio_info['duration'],
                    confidence=0.0,
                    percentage=100.0
                )
            ]
        
        # Step 6: Format results
        logger.info(f"📦 Step 6/6: Formatting results...")
        
        transcript_data = [seg.to_dict() for seg in merged_segments]
        speakers_data = [spk.to_dict() for spk in speakers_info]
        
        result_metadata = {
            'audioDuration': audio_info['duration'],
            'audioSampleRate': audio_info['sample_rate'],
            'audioChannels': audio_info['channels'],
            'audioFormat': audio_info.get('format', 'unknown'),
            'totalSegments': len(transcript_data),
            'uniqueSpeakers': len(speakers_data),
            'whisperModel': settings.WHISPER_MODEL,
            'whisperLanguage': settings.WHISPER_LANGUAGE,
            'diarizationEnabled': settings.DIARIZATION_ENABLED,
            'diarizationModel': settings.DIARIZATION_MODEL if settings.DIARIZATION_ENABLED else None,
        }
        
        logger.info(f"✅ Processing complete:")
        logger.info(f"   - Transcript segments: {len(transcript_data)}")
        logger.info(f"   - Unique speakers: {len(speakers_data)}")
        logger.info(f"   - Audio duration: {audio_info['duration']:.2f}s")
        
        return {
            'transcript': transcript_data,
            'speakers': speakers_data,
            'metadata': result_metadata
        }
    
    def _cleanup_temp_files(self, temp_files: list):
        """Clean up temporary files"""
        if not settings.CLEANUP_TEMP_FILES:
            logger.debug("Temp file cleanup disabled")
            return
        
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temp file {file_path}: {e}")
    
    def stop(self):
        """Stop the worker gracefully"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🛑 Stopping {self.worker_name}...")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   - Jobs succeeded: {self.jobs_succeeded}")
        logger.info(f"   - Jobs failed: {self.jobs_failed}")
        logger.info(f"   - Jobs total: {self.jobs_processed}")
        logger.info("=" * 80)
        self.running = False


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"⚠️ Received signal {signum}")
    sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start worker
    worker = ASRWorker()
    worker.start()


if __name__ == "__main__":
    main()
