"""
Diarization Service - Speaker separation and identification

Handles speaker diarization using pyannote.audio to identify who spoke when.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add PyTorch DLL directory for cuDNN (MUST be before pyannote import)
try:
    import torch
    from pathlib import Path
    torch_lib = Path(torch.__file__).parent / "lib"
    os.add_dll_directory(str(torch_lib))
except Exception:
    pass  # If fails, continue anyway

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError:
    Pipeline = None
    Annotation = None
    Segment = None

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import DiarizationError, ModelLoadError
from services.asr_service import TranscriptSegment

logger = get_logger(__name__)


@dataclass
class DiarizationSegment:
    """Represents a speaker segment with timing"""
    
    speaker_label: str
    start: float
    end: float
    duration: float
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'speakerLabel': self.speaker_label,
            'start': round(self.start, 2),
            'end': round(self.end, 2),
            'duration': round(self.duration, 2),
            'confidence': round(self.confidence, 3)
        }
    
    def overlaps_with(self, start: float, end: float) -> bool:
        """Check if this segment overlaps with given time range"""
        return not (self.end <= start or self.start >= end)
    
    def overlap_duration(self, start: float, end: float) -> float:
        """Calculate overlap duration with given time range"""
        if not self.overlaps_with(start, end):
            return 0.0
        
        overlap_start = max(self.start, start)
        overlap_end = min(self.end, end)
        return overlap_end - overlap_start


@dataclass
class SpeakerInfo:
    """Aggregated speaker information"""
    
    speaker_label: str
    segment_count: int
    total_duration: float
    confidence: float
    percentage: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'aiSpeakerLabel': self.speaker_label,
            'segmentCount': self.segment_count,
            'totalDuration': round(self.total_duration, 2),
            'confidence': round(self.confidence, 3),
            'percentage': round(self.percentage, 2)
        }


@dataclass
class TranscriptWithSpeaker:
    """Transcript segment with assigned speaker"""
    
    id: int
    text: str
    start: float
    end: float
    speaker_label: str
    confidence: float
    transcript_confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text.strip(),
            'start': round(self.start, 2),
            'end': round(self.end, 2),
            'duration': round(self.end - self.start, 2),
            'speakerLabel': self.speaker_label,
            'confidence': round(self.confidence, 3),
            'transcriptConfidence': round(self.transcript_confidence, 3)
        }


class DiarizationService:
    """Service for speaker diarization using pyannote.audio"""
    
    def __init__(self):
        """Initialize diarization service"""
        self.pipeline = None
        self.model_name = settings.DIARIZATION_MODEL
        self.min_speakers = settings.MIN_SPEAKERS
        self.max_speakers = settings.MAX_SPEAKERS
        self.hf_token = settings.HUGGINGFACE_TOKEN
        
        # Pipeline cache
        self._pipeline_loaded = False
        
        logger.info("✅ Diarization service initialized")
        logger.info(f"   - Model: {self.model_name}")
        logger.info(f"   - Min speakers: {self.min_speakers}")
        logger.info(f"   - Max speakers: {self.max_speakers}")
        logger.info(f"   - HF token: {'configured' if self.hf_token else 'not set'}")
    
    def load_pipeline(self) -> Pipeline:
        """
        Load pyannote diarization pipeline (lazy loading)
        
        Returns:
            Pipeline instance
            
        Raises:
            ModelLoadError: If pipeline loading fails
        """
        if self._pipeline_loaded and self.pipeline is not None:
            logger.debug("Using cached diarization pipeline")
            return self.pipeline
        
        try:
            if Pipeline is None:
                raise ModelLoadError(
                    "pyannote.audio not installed",
                    model_name=self.model_name,
                    details={
                        "install_hint": "pip install pyannote.audio"
                    }
                )
            
            if not self.hf_token:
                logger.warning(
                    "⚠️ HUGGINGFACE_TOKEN not set. "
                    "You may need to accept model conditions at: "
                    "https://huggingface.co/pyannote/speaker-diarization"
                )
            
            logger.info(f"🔄 Loading diarization pipeline: {self.model_name}...")
            start_time = time.time()
            
            # Fix PyTorch 2.6+ weights_only issue
            # Temporarily disable weights_only check for pyannote models
            import torch
            import warnings
            original_torch_load = torch.load
            
            def patched_torch_load(*args, **kwargs):
                """Patch torch.load to use weights_only=False for compatibility"""
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    return original_torch_load(*args, **kwargs)
            
            # Apply patch
            torch.load = patched_torch_load
            
            try:
                # Load pipeline
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token
                )
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
            
            # Move to device if CUDA available (always try for diarization)
            try:
                import torch
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                    logger.info("   - Using CUDA for diarization")
                else:
                    logger.info("   - Using CPU for diarization (CUDA not available)")
            except Exception as e:
                logger.warning(f"   - Failed to use CUDA, using CPU: {e}")
            
            load_time = time.time() - start_time
            self._pipeline_loaded = True
            
            logger.info(f"✅ Diarization pipeline loaded in {load_time:.2f}s")
            
            return self.pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to load diarization pipeline: {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load diarization pipeline: {str(e)}",
                model_name=self.model_name,
                details={
                    "hint": "Check HUGGINGFACE_TOKEN and accept model license"
                }
            )
    
    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers (default: settings.MIN_SPEAKERS)
            max_speakers: Maximum number of speakers (default: settings.MAX_SPEAKERS)
            
        Returns:
            List of DiarizationSegment objects
            
        Raises:
            DiarizationError: If diarization fails
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                raise DiarizationError(
                    f"Audio file not found: {audio_path}",
                    audio_path=audio_path,
                    model=self.model_name
                )
            
            # Use configured values if not specified
            if min_speakers is None:
                min_speakers = self.min_speakers
            if max_speakers is None:
                max_speakers = self.max_speakers
            
            logger.info(f"👥 Starting speaker diarization...")
            logger.info(f"   - Audio: {os.path.basename(audio_path)}")
            logger.info(f"   - Min speakers: {min_speakers}")
            logger.info(f"   - Max speakers: {max_speakers}")
            
            # Load pipeline (lazy loading)
            pipeline = self.load_pipeline()
            
            # Perform diarization
            start_time = time.time()
            
            diarization = pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            diarization_time = time.time() - start_time
            
            # Convert pyannote output to DiarizationSegment list
            segments = []
            speaker_stats = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Format speaker label as SPEAKER_00, SPEAKER_01, etc.
                speaker_label = self._format_speaker_label(speaker)
                
                segment = DiarizationSegment(
                    speaker_label=speaker_label,
                    start=turn.start,
                    end=turn.end,
                    duration=turn.end - turn.start,
                    confidence=1.0  # pyannote doesn't provide confidence per segment
                )
                
                segments.append(segment)
                
                # Track speaker statistics
                if speaker_label not in speaker_stats:
                    speaker_stats[speaker_label] = {
                        'count': 0,
                        'total_duration': 0.0
                    }
                
                speaker_stats[speaker_label]['count'] += 1
                speaker_stats[speaker_label]['total_duration'] += segment.duration
            
            # Calculate total duration
            total_duration = max(seg.end for seg in segments) if segments else 0
            
            # Get unique speakers
            unique_speakers = len(speaker_stats)
            
            logger.info(f"✅ Diarization complete")
            logger.info(f"   - Segments: {len(segments)}")
            logger.info(f"   - Unique speakers: {unique_speakers}")
            logger.info(f"   - Total duration: {total_duration:.2f}s")
            logger.info(f"   - Processing time: {diarization_time:.2f}s")
            logger.info(f"   - Speed: {total_duration / diarization_time:.2f}x realtime")
            
            # Log speaker breakdown
            for speaker, stats in sorted(speaker_stats.items()):
                percentage = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
                logger.info(
                    f"   - {speaker}: {stats['count']} segments, "
                    f"{stats['total_duration']:.2f}s ({percentage:.1f}%)"
                )
            
            return segments
            
        except DiarizationError:
            raise
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}", exc_info=True)
            raise DiarizationError(
                f"Diarization failed: {str(e)}",
                audio_path=audio_path,
                model=self.model_name
            )
    
    def merge_with_transcript(
        self,
        transcript_segments: List[TranscriptSegment],
        diarization_segments: List[DiarizationSegment],
        overlap_threshold: float = 0.5
    ) -> List[TranscriptWithSpeaker]:
        """
        Merge transcript with diarization to assign speakers to each transcript segment
        
        Algorithm:
        - For each transcript segment, find overlapping diarization segments
        - Assign speaker with maximum overlap duration
        - If overlap < threshold, mark as UNKNOWN
        
        Args:
            transcript_segments: List of transcript segments from ASR
            diarization_segments: List of diarization segments
            overlap_threshold: Minimum overlap ratio to assign speaker (0-1)
            
        Returns:
            List of TranscriptWithSpeaker objects
            
        Example:
            transcript_segments = asr.transcribe(audio)
            diarization_segments = diarization.diarize(audio)
            merged = diarization.merge_with_transcript(transcript_segments, diarization_segments)
        """
        logger.info(f"🔗 Merging transcript with diarization...")
        logger.info(f"   - Transcript segments: {len(transcript_segments)}")
        logger.info(f"   - Diarization segments: {len(diarization_segments)}")
        logger.info(f"   - Overlap threshold: {overlap_threshold}")
        
        merged_segments = []
        unknown_count = 0
        
        for ts in transcript_segments:
            # Find overlapping diarization segments
            overlapping = self._find_overlapping_segments(
                ts.start,
                ts.end,
                diarization_segments
            )
            
            if not overlapping:
                # No overlap - assign UNKNOWN
                speaker_label = "UNKNOWN"
                speaker_confidence = 0.0
                unknown_count += 1
            else:
                # Calculate overlap for each speaker
                speaker_overlaps = {}
                
                for diaz_seg in overlapping:
                    overlap_duration = diaz_seg.overlap_duration(ts.start, ts.end)
                    
                    if diaz_seg.speaker_label not in speaker_overlaps:
                        speaker_overlaps[diaz_seg.speaker_label] = 0.0
                    
                    speaker_overlaps[diaz_seg.speaker_label] += overlap_duration
                
                # Get speaker with maximum overlap
                max_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])
                speaker_label = max_speaker[0]
                overlap_duration = max_speaker[1]
                
                # Calculate confidence based on overlap ratio
                segment_duration = ts.end - ts.start
                overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0
                
                # Check if overlap meets threshold
                if overlap_ratio < overlap_threshold:
                    speaker_label = "UNKNOWN"
                    speaker_confidence = overlap_ratio
                    unknown_count += 1
                else:
                    speaker_confidence = overlap_ratio
            
            # Create merged segment
            merged = TranscriptWithSpeaker(
                id=ts.id,
                text=ts.text,
                start=ts.start,
                end=ts.end,
                speaker_label=speaker_label,
                confidence=speaker_confidence,
                transcript_confidence=ts.confidence
            )
            
            merged_segments.append(merged)
        
        logger.info(f"✅ Merge complete")
        logger.info(f"   - Merged segments: {len(merged_segments)}")
        logger.info(f"   - Unknown speaker: {unknown_count} segments")
        
        # Log speaker distribution
        speaker_counts = {}
        for seg in merged_segments:
            if seg.speaker_label not in speaker_counts:
                speaker_counts[seg.speaker_label] = 0
            speaker_counts[seg.speaker_label] += 1
        
        for speaker, count in sorted(speaker_counts.items()):
            percentage = (count / len(merged_segments) * 100) if merged_segments else 0
            logger.info(f"   - {speaker}: {count} segments ({percentage:.1f}%)")
        
        return merged_segments
    
    def get_speaker_info(
        self,
        merged_segments: List[TranscriptWithSpeaker]
    ) -> List[SpeakerInfo]:
        """
        Extract aggregated speaker information from merged segments
        
        Args:
            merged_segments: List of merged transcript segments
            
        Returns:
            List of SpeakerInfo objects
        """
        speaker_data = {}
        total_duration = 0.0
        
        # Aggregate by speaker
        for seg in merged_segments:
            if seg.speaker_label == "UNKNOWN":
                continue
            
            duration = seg.end - seg.start
            total_duration += duration
            
            if seg.speaker_label not in speaker_data:
                speaker_data[seg.speaker_label] = {
                    'count': 0,
                    'duration': 0.0,
                    'confidences': []
                }
            
            speaker_data[seg.speaker_label]['count'] += 1
            speaker_data[seg.speaker_label]['duration'] += duration
            speaker_data[seg.speaker_label]['confidences'].append(seg.confidence)
        
        # Build SpeakerInfo list
        speaker_info_list = []
        
        for speaker_label, data in sorted(speaker_data.items()):
            avg_confidence = sum(data['confidences']) / len(data['confidences'])
            percentage = (data['duration'] / total_duration * 100) if total_duration > 0 else 0
            
            info = SpeakerInfo(
                speaker_label=speaker_label,
                segment_count=data['count'],
                total_duration=data['duration'],
                confidence=avg_confidence,
                percentage=percentage
            )
            
            speaker_info_list.append(info)
        
        # Sort by duration (descending)
        speaker_info_list.sort(key=lambda x: x.total_duration, reverse=True)
        
        return speaker_info_list
    
    def _find_overlapping_segments(
        self,
        start: float,
        end: float,
        segments: List[DiarizationSegment]
    ) -> List[DiarizationSegment]:
        """Find all segments that overlap with given time range"""
        overlapping = []
        
        for seg in segments:
            if seg.overlaps_with(start, end):
                overlapping.append(seg)
        
        return overlapping
    
    def _format_speaker_label(self, speaker: str) -> str:
        """
        Format speaker label to consistent format: SPEAKER_00, SPEAKER_01, etc.
        
        Args:
            speaker: Raw speaker label from pyannote (e.g., "SPEAKER_0", "speaker1")
            
        Returns:
            Formatted label (e.g., "SPEAKER_00")
        """
        # Extract number from speaker label
        import re
        match = re.search(r'(\d+)', speaker)
        
        if match:
            speaker_num = int(match.group(1))
            return f"SPEAKER_{speaker_num:02d}"
        else:
            # Fallback: use original label
            return speaker.upper()
    
    def unload_pipeline(self) -> None:
        """Unload pipeline from memory"""
        if self.pipeline is not None:
            logger.info("🗑️ Unloading diarization pipeline from memory")
            del self.pipeline
            self.pipeline = None
            self._pipeline_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()


class MockDiarizationService:
    """Mock diarization service for testing without pyannote"""
    
    def __init__(self):
        logger.warning("⚠️ Using MOCK Diarization Service (for testing only)")
    
    def load_pipeline(self):
        logger.info("📦 Mock: Pipeline loaded")
        return self
    
    def diarize(self, audio_path: str, **kwargs) -> List[DiarizationSegment]:
        """Return mock diarization segments"""
        logger.info(f"👥 Mock: Diarizing {audio_path}")
        
        # Generate 3 speakers with segments
        mock_segments = [
            DiarizationSegment("SPEAKER_00", 0.0, 4.5, 4.5, 1.0),
            DiarizationSegment("SPEAKER_01", 4.5, 8.2, 3.7, 1.0),
            DiarizationSegment("SPEAKER_00", 8.2, 12.0, 3.8, 1.0),
            DiarizationSegment("SPEAKER_02", 12.0, 16.5, 4.5, 1.0),
            DiarizationSegment("SPEAKER_01", 16.5, 20.0, 3.5, 1.0),
        ]
        
        logger.info(f"✅ Mock: Generated {len(mock_segments)} segments with 3 speakers")
        return mock_segments
    
    def merge_with_transcript(self, transcript_segments, diarization_segments, **kwargs):
        """Mock merge"""
        merged = []
        for ts in transcript_segments:
            # Simple assignment based on segment ID
            speaker_idx = ts.id % 3
            merged.append(TranscriptWithSpeaker(
                id=ts.id,
                text=ts.text,
                start=ts.start,
                end=ts.end,
                speaker_label=f"SPEAKER_{speaker_idx:02d}",
                confidence=0.95,
                transcript_confidence=ts.confidence
            ))
        return merged
    
    def get_speaker_info(self, merged_segments):
        """Mock speaker info"""
        return [
            SpeakerInfo("SPEAKER_00", 15, 120.5, 0.95, 40.0),
            SpeakerInfo("SPEAKER_01", 12, 90.3, 0.92, 30.0),
            SpeakerInfo("SPEAKER_02", 18, 90.2, 0.89, 30.0)
        ]


# Singleton instance
_diarization_service_instance = None


def get_diarization_service(use_mock: bool = None) -> DiarizationService:
    """
    Get DiarizationService singleton instance
    
    Args:
        use_mock: Use mock service (default: settings.MOCK_ASR)
        
    Returns:
        DiarizationService or MockDiarizationService instance
    """
    global _diarization_service_instance
    
    if _diarization_service_instance is None:
        if use_mock is None:
            use_mock = settings.MOCK_ASR  # Reuse same flag
        
        if use_mock:
            _diarization_service_instance = MockDiarizationService()
        else:
            _diarization_service_instance = DiarizationService()
    
    return _diarization_service_instance
