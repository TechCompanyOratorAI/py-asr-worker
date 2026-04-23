"""
ASR Service - Automatic Speech Recognition using Whisper

Handles speech-to-text transcription using faster-whisper model.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    WhisperModel = None
    BatchedInferencePipeline = None

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import ASRProcessingError, ModelLoadError
from utils.validators import AudioValidator

logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with timing"""
    
    id: int
    text: str
    start: float
    end: float
    confidence: float
    no_speech_prob: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text.strip(),
            'start': round(self.start, 2),
            'end': round(self.end, 2),
            'duration': round(self.end - self.start, 2),
            'confidence': round(self.confidence, 3),
            'noSpeechProb': round(self.no_speech_prob, 3)
        }


class ASRService:
    """Service for Automatic Speech Recognition using Whisper"""
    
    def __init__(self):
        """Initialize ASR service"""
        self.model = None
        self.batched_model = None  # ⚡ BatchedInferencePipeline
        self.model_name = settings.WHISPER_MODEL
        self.language = settings.WHISPER_LANGUAGE
        self.device = settings.WHISPER_DEVICE
        self.compute_type = settings.WHISPER_COMPUTE_TYPE
        
        # Model cache
        self._model_loaded = False
        
        logger.info("✅ ASR service initialized")
        logger.info(f"   - Model: {self.model_name}")
        logger.info(f"   - Language: {self.language}")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Compute type: {self.compute_type}")
        logger.info(f"   - Batched inference: {BatchedInferencePipeline is not None}")
    
    def load_model(self) -> WhisperModel:
        """
        Load Whisper model (lazy loading)
        
        Returns:
            WhisperModel instance
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model_loaded and self.model is not None:
            logger.debug("Using cached Whisper model")
            return self.batched_model if self.batched_model else self.model
        
        try:
            if WhisperModel is None:
                raise ModelLoadError(
                    "faster-whisper not installed",
                    model_name=self.model_name,
                    details={
                        "install_hint": "pip install faster-whisper"
                    }
                )
            
            logger.info(f"🔄 Loading Whisper model: {self.model_name}...")
            start_time = time.time()
            
            # Load base model
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache
                local_files_only=False,
                cpu_threads=4,   # ⚡ Parallel CPU threads for preprocessing
                num_workers=1,   # ⚡ CTranslate2 async workers
            )
            
            # ⚡ Wrap in BatchedInferencePipeline for ~4-6x faster transcription
            if BatchedInferencePipeline is not None:
                self.batched_model = BatchedInferencePipeline(model=self.model)
                logger.info("   ⚡ BatchedInferencePipeline enabled")
            else:
                self.batched_model = None
                logger.warning("   ⚠️ BatchedInferencePipeline not available, using sequential")
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"✅ Whisper model loaded in {load_time:.2f}s")
            
            return self.batched_model if self.batched_model else self.model
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load Whisper model: {str(e)}",
                model_name=self.model_name,
                details={
                    "device": self.device,
                    "compute_type": self.compute_type
                }
            )
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        temperature: float = 0.0
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio file to text with timestamps
        
        Args:
            audio_path: Path to audio file (preferably 16kHz mono WAV)
            language: Language code (default: settings.WHISPER_LANGUAGE)
            beam_size: Beam size for decoding (higher = more accurate, slower)
            vad_filter: Enable Voice Activity Detection (removes silence)
            temperature: Sampling temperature (0 = greedy, >0 = sampling)
            
        Returns:
            List of TranscriptSegment objects
            
        Raises:
            ASRProcessingError: If transcription fails
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                raise ASRProcessingError(
                    f"Audio file not found: {audio_path}",
                    audio_path=audio_path,
                    engine="whisper"
                )
            
            # Get audio duration for logging
            file_size = os.path.getsize(audio_path)
            
            # Use configured language if not specified
            if language is None:
                language = self.language
            
            logger.info(f"🎤 Starting ASR transcription...")
            logger.info(f"   - Audio: {os.path.basename(audio_path)}")
            logger.info(f"   - Size: {self._format_size(file_size)}")
            logger.info(f"   - Language: {language}")
            logger.info(f"   - Beam size: {beam_size}")
            logger.info(f"   - VAD filter: {vad_filter}")
            
            # Load model (lazy loading)
            model = self.load_model()
            
            # Re-enable TF32 for better GPU performance
            # (pyannote disables it globally, affecting all torch operations)
            try:
                import torch
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            
            # Transcribe
            start_time = time.time()
            
            # Whisper accuracy params driven by settings (tuned for noisy audio)
            no_speech_threshold = settings.WHISPER_NO_SPEECH_THRESHOLD
            log_prob_threshold  = settings.WHISPER_LOG_PROB_THRESHOLD
            compression_ratio   = settings.WHISPER_COMPRESSION_RATIO_THRESHOLD
            condition_on_prev   = settings.WHISPER_CONDITION_ON_PREV_TEXT
            
            # Fallback temperatures: if greedy (0.0) confidence is low,
            # Whisper will retry with higher temperatures — reduces hallucination on noise.
            temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if temperature == 0.0 else temperature
            
            logger.info(f"   - no_speech_threshold: {no_speech_threshold}")
            logger.info(f"   - log_prob_threshold: {log_prob_threshold}")
            logger.info(f"   - condition_on_prev_text: {condition_on_prev}")
            logger.info(f"   - temperatures: {temperatures}")
            
            # ⚡ Use BatchedInferencePipeline for ~4-6x faster transcription
            if self.batched_model is not None:
                logger.info("   ⚡ Using batched inference (batch_size=4)")
                segments, info = self.batched_model.transcribe(
                    audio=audio_path,
                    language=language,
                    beam_size=beam_size,
                    batch_size=4,           # ⚡ RTX 3060 6GB: safe batch size
                    vad_filter=vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,  # shorter window catches more speech
                        speech_pad_ms=200,
                    ),
                    temperature=temperatures,
                    word_timestamps=False,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    compression_ratio_threshold=compression_ratio,
                )
            else:
                # Fallback to sequential transcription
                logger.info("   🔄 Using sequential inference")
                segments, info = model.transcribe(
                    audio=audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=200,
                    ),
                    temperature=temperatures,
                    word_timestamps=False,
                    condition_on_previous_text=condition_on_prev,
                    compression_ratio_threshold=compression_ratio,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    chunk_length=30,
                )
            
            # Convert generator to list and format segments
            transcript_segments = []
            segment_count = 0
            
            for segment in segments:
                segment_count += 1
                
                # Calculate average log probability as confidence
                confidence = self._calculate_confidence(segment)
                
                # Create TranscriptSegment
                ts = TranscriptSegment(
                    id=segment.id,
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    confidence=confidence,
                    no_speech_prob=segment.no_speech_prob
                )
                
                transcript_segments.append(ts)
                
                # Log first few segments
                if segment_count <= 3:
                    logger.debug(
                        f"Segment {segment.id}: [{segment.start:.2f}s - {segment.end:.2f}s] "
                        f"{segment.text[:50]}..."
                    )
            
            transcription_time = time.time() - start_time
            
            # Calculate statistics
            total_duration = transcript_segments[-1].end if transcript_segments else 0
            avg_confidence = sum(s.confidence for s in transcript_segments) / len(transcript_segments) if transcript_segments else 0
            
            logger.info(f"✅ Transcription complete")
            logger.info(f"   - Segments: {len(transcript_segments)}")
            logger.info(f"   - Audio duration: {total_duration:.2f}s")
            logger.info(f"   - Processing time: {transcription_time:.2f}s")
            logger.info(f"   - Speed: {total_duration / transcription_time:.2f}x realtime")
            logger.info(f"   - Avg confidence: {avg_confidence:.3f}")
            logger.info(f"   - Detected language: {info.language}")
            logger.info(f"   - Language probability: {info.language_probability:.3f}")
            
            return transcript_segments
            
        except ASRProcessingError:
            raise
        except Exception as e:
            logger.error(f"❌ ASR transcription failed: {e}", exc_info=True)
            raise ASRProcessingError(
                f"ASR transcription failed: {str(e)}",
                audio_path=audio_path,
                engine="whisper",
                details={
                    "language": language,
                    "model": self.model_name
                }
            )
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None
    ) -> Dict[str, List[TranscriptSegment]]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            language: Language code
            
        Returns:
            Dictionary mapping audio_path -> transcript_segments
        """
        results = {}
        
        logger.info(f"🎤 Starting batch transcription for {len(audio_paths)} files")
        
        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"Processing {i}/{len(audio_paths)}: {os.path.basename(audio_path)}")
            
            try:
                segments = self.transcribe(audio_path, language=language)
                results[audio_path] = segments
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                results[audio_path] = []
        
        logger.info(f"✅ Batch transcription complete: {len(results)} files processed")
        
        return results
    
    def get_full_transcript(
        self,
        segments: List[TranscriptSegment],
        include_timestamps: bool = False
    ) -> str:
        """
        Get full transcript as single text
        
        Args:
            segments: List of transcript segments
            include_timestamps: Include timestamps in output
            
        Returns:
            Full transcript text
        """
        if include_timestamps:
            lines = [
                f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text.strip()}"
                for seg in segments
            ]
            return "\n".join(lines)
        else:
            return " ".join(seg.text.strip() for seg in segments)
    
    def filter_low_confidence_segments(
        self,
        segments: List[TranscriptSegment],
        min_confidence: float = 0.5
    ) -> List[TranscriptSegment]:
        """
        Filter out segments with low confidence
        
        Args:
            segments: List of transcript segments
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            Filtered segments
        """
        filtered = [seg for seg in segments if seg.confidence >= min_confidence]
        
        if len(filtered) < len(segments):
            logger.warning(
                f"⚠️ Filtered out {len(segments) - len(filtered)} "
                f"low-confidence segments (< {min_confidence})"
            )
        
        return filtered
    
    def format_segments_for_api(
        self,
        segments: List[TranscriptSegment]
    ) -> List[Dict]:
        """
        Format segments for API response
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of segment dictionaries
        """
        return [seg.to_dict() for seg in segments]
    
    def _calculate_confidence(self, segment) -> float:
        """
        Calculate confidence score from segment
        
        Uses average log probability and no_speech_prob
        
        Args:
            segment: Whisper segment object
            
        Returns:
            Confidence score (0-1)
        """
        # Get average log probability
        avg_logprob = getattr(segment, 'avg_logprob', -1.0)
        
        # Convert log probability to linear scale (approximate)
        # avg_logprob ranges from -1 (high confidence) to -3 (low confidence)
        confidence = min(1.0, max(0.0, 1.0 + (avg_logprob / 3.0)))
        
        # Adjust by no_speech probability
        no_speech_prob = getattr(segment, 'no_speech_prob', 0.0)
        confidence = confidence * (1.0 - no_speech_prob)
        
        return confidence
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            logger.info("🗑️ Unloading Whisper model from memory")
            del self.model
            self.model = None
            self._model_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        return [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr',
            'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi',
            'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no',
            'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk',
            'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk',
            'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
            'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
            'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo',
            'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl',
            'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]


class MockASRService:
    """Mock ASR service for testing without Whisper model"""
    
    def __init__(self):
        logger.warning("⚠️ Using MOCK ASR Service (for testing only)")
    
    def load_model(self):
        logger.info("📦 Mock: Model loaded")
        return self
    
    def transcribe(self, audio_path: str, **kwargs) -> List[TranscriptSegment]:
        """Return mock transcript"""
        logger.info(f"🎤 Mock: Transcribing {audio_path}")
        
        # Generate mock segments
        mock_segments = [
            TranscriptSegment(
                id=0,
                text="Xin chào các bạn, hôm nay tôi sẽ trình bày về chủ đề AI.",
                start=0.0,
                end=4.5,
                confidence=0.95
            ),
            TranscriptSegment(
                id=1,
                text="Trí tuệ nhân tạo đang thay đổi thế giới.",
                start=4.5,
                end=8.2,
                confidence=0.92
            ),
            TranscriptSegment(
                id=2,
                text="Chúng ta sẽ cùng tìm hiểu các ứng dụng thực tế.",
                start=8.2,
                end=12.0,
                confidence=0.89
            )
        ]
        
        logger.info(f"✅ Mock: Generated {len(mock_segments)} segments")
        return mock_segments
    
    def format_segments_for_api(self, segments):
        return [seg.to_dict() for seg in segments]


# Singleton instance
_asr_service_instance = None


def get_asr_service(use_mock: bool = None) -> ASRService:
    """
    Get ASRService singleton instance
    
    Args:
        use_mock: Use mock service (default: settings.MOCK_ASR)
        
    Returns:
        ASRService or MockASRService instance
    """
    global _asr_service_instance
    
    if _asr_service_instance is None:
        if use_mock is None:
            use_mock = settings.MOCK_ASR
        
        if use_mock:
            _asr_service_instance = MockASRService()
        else:
            _asr_service_instance = ASRService()
    
    return _asr_service_instance
