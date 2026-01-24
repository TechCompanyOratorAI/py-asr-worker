"""
Audio Processor Service - Audio normalization and validation

Handles audio file processing using FFmpeg for ASR preparation.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import AudioProcessingError, AudioValidationError
from utils.validators import AudioValidator
from utils.helpers import ensure_directory

logger = get_logger(__name__)


class AudioProcessor:
    """Service for audio file processing and normalization"""
    
    def __init__(self):
        """Initialize audio processor"""
        self._verify_ffmpeg()
        logger.info("✅ Audio processor initialized")
    
    def normalize_audio(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> str:
        """
        Normalize audio file for ASR processing
        
        Converts audio to:
        - Sample rate: 16kHz (configurable)
        - Channels: Mono (1 channel)
        - Format: WAV (PCM 16-bit)
        - Normalized volume
        
        Args:
            input_path: Path to input audio file
            output_dir: Output directory (default: same as input)
            sample_rate: Target sample rate in Hz (default: 16000)
            channels: Number of channels (default: 1 for mono)
            
        Returns:
            Path to normalized audio file
            
        Raises:
            AudioProcessingError: If normalization fails
        """
        try:
            # Validate input file exists
            if not os.path.exists(input_path):
                raise AudioProcessingError(
                    f"Input audio file not found: {input_path}",
                    audio_path=input_path
                )
            
            # Determine output path
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
            
            ensure_directory(output_dir)
            
            # Generate output filename
            input_filename = os.path.basename(input_path)
            filename_without_ext = os.path.splitext(input_filename)[0]
            output_path = os.path.join(output_dir, f"{filename_without_ext}_normalized.wav")
            
            logger.info(f"🎵 Normalizing audio...")
            logger.info(f"   - Input: {input_path}")
            logger.info(f"   - Output: {output_path}")
            logger.info(f"   - Sample rate: {sample_rate} Hz")
            logger.info(f"   - Channels: {channels} (mono)")
            
            # Get input audio info
            input_info = self.get_audio_info(input_path)
            logger.info(f"   - Input duration: {input_info['duration']:.2f}s")
            logger.info(f"   - Input format: {input_info.get('format', 'unknown')}")
            
            # Build FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,           # Input file
                '-ar', str(sample_rate),    # Sample rate
                '-ac', str(channels),       # Channels (1 = mono)
                '-c:a', 'pcm_s16le',       # Codec: PCM 16-bit little-endian
                '-y',                       # Overwrite output file
                output_path                 # Output file
            ]
            
            # Add audio normalization filter
            # Uses loudnorm filter for EBU R128 loudness normalization
            if settings.DEBUG:
                logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Execute FFmpeg
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.PROCESSING_TIMEOUT
            )
            
            if result.returncode != 0:
                error_output = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"❌ FFmpeg error: {error_output}")
                raise AudioProcessingError(
                    "FFmpeg normalization failed",
                    audio_path=input_path,
                    details={
                        "return_code": result.returncode,
                        "error": error_output[:500]  # First 500 chars
                    }
                )
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise AudioProcessingError(
                    "Normalized audio file not created",
                    audio_path=input_path,
                    details={"expected_output": output_path}
                )
            
            # Get output info
            output_info = self.get_audio_info(output_path)
            output_size = os.path.getsize(output_path)
            
            logger.info(f"✅ Audio normalized successfully")
            logger.info(f"   - Duration: {output_info['duration']:.2f}s")
            logger.info(f"   - Sample rate: {output_info['sample_rate']} Hz")
            logger.info(f"   - Channels: {output_info['channels']}")
            logger.info(f"   - Size: {self._format_size(output_size)}")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                f"Audio normalization timed out after {settings.PROCESSING_TIMEOUT}s",
                audio_path=input_path
            )
        except AudioProcessingError:
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during normalization: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Unexpected error during normalization: {str(e)}",
                audio_path=input_path
            )
    
    def get_audio_info(self, audio_path: str) -> Dict[str, any]:
        """
        Extract audio file metadata using FFprobe
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata:
            - duration: Duration in seconds (float)
            - sample_rate: Sample rate in Hz (int)
            - channels: Number of channels (int)
            - codec: Audio codec name (str)
            - bitrate: Bitrate in bits/sec (int)
            - format: Container format (str)
            - size: File size in bytes (int)
            
        Raises:
            AudioProcessingError: If metadata extraction fails
        """
        try:
            if not os.path.exists(audio_path):
                raise AudioProcessingError(
                    f"Audio file not found: {audio_path}",
                    audio_path=audio_path
                )
            
            # FFprobe command
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'quiet',              # Quiet mode
                '-print_format', 'json',    # JSON output
                '-show_format',             # Show format info
                '-show_streams',            # Show stream info
                audio_path
            ]
            
            result = subprocess.run(
                ffprobe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30  # 30 seconds timeout
            )
            
            if result.returncode != 0:
                error_output = result.stderr.decode('utf-8', errors='ignore')
                raise AudioProcessingError(
                    "FFprobe failed to extract audio info",
                    audio_path=audio_path,
                    details={"error": error_output[:200]}
                )
            
            # Parse JSON output
            probe_data = json.loads(result.stdout.decode('utf-8'))
            
            # Extract format info
            format_info = probe_data.get('format', {})
            
            # Extract audio stream info (first audio stream)
            audio_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise AudioProcessingError(
                    "No audio stream found in file",
                    audio_path=audio_path
                )
            
            # Build metadata dictionary
            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format': format_info.get('format_name', 'unknown'),
                'size': int(format_info.get('size', os.path.getsize(audio_path)))
            }
            
            return metadata
            
        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                "FFprobe timed out",
                audio_path=audio_path
            )
        except json.JSONDecodeError as e:
            raise AudioProcessingError(
                f"Failed to parse FFprobe output: {e}",
                audio_path=audio_path
            )
        except Exception as e:
            logger.error(f"❌ Error extracting audio info: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to extract audio info: {str(e)}",
                audio_path=audio_path
            )
    
    def validate_audio(self, audio_path: str) -> Tuple[bool, list]:
        """
        Validate audio file for ASR processing
        
        Checks:
        - File exists and readable
        - Valid audio format
        - Duration within limits
        - File size within limits
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, errors_list)
            
        Example:
            is_valid, errors = processor.validate_audio("audio.mp3")
            if not is_valid:
                for error in errors:
                    print(f"Validation error: {error}")
        """
        errors = []
        
        try:
            # Check file exists
            if not os.path.exists(audio_path):
                errors.append(f"File not found: {audio_path}")
                return False, errors
            
            # Validate file path (extension)
            try:
                AudioValidator.validate_file_path(audio_path)
            except Exception as e:
                errors.append(f"Invalid file format: {str(e)}")
            
            # Get file size
            file_size = os.path.getsize(audio_path)
            
            # Validate file size
            try:
                AudioValidator.validate_file_size(file_size, audio_path)
            except Exception as e:
                errors.append(f"Invalid file size: {str(e)}")
            
            # Get audio metadata
            try:
                info = self.get_audio_info(audio_path)
            except Exception as e:
                errors.append(f"Failed to read audio metadata: {str(e)}")
                return False, errors
            
            # Validate duration
            try:
                AudioValidator.validate_duration(
                    info['duration'],
                    settings.MAX_AUDIO_LENGTH
                )
            except Exception as e:
                errors.append(f"Invalid duration: {str(e)}")
            
            # Additional checks
            if info['sample_rate'] == 0:
                errors.append("Invalid sample rate: 0 Hz")
            
            if info['channels'] == 0:
                errors.append("Invalid channel count: 0")
            
            # Return validation result
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info(f"✅ Audio validation passed")
                logger.info(f"   - Duration: {info['duration']:.2f}s")
                logger.info(f"   - Sample rate: {info['sample_rate']} Hz")
                logger.info(f"   - Channels: {info['channels']}")
            else:
                logger.warning(f"⚠️ Audio validation failed with {len(errors)} error(s)")
                for error in errors:
                    logger.warning(f"   - {error}")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"❌ Error during audio validation: {e}", exc_info=True)
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def convert_to_wav(
        self,
        input_path: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Convert audio file to WAV format (without normalization)
        
        Args:
            input_path: Path to input audio file
            output_dir: Output directory (default: same as input)
            
        Returns:
            Path to WAV file
            
        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
            
            ensure_directory(output_dir)
            
            filename_without_ext = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"{filename_without_ext}.wav")
            
            logger.info(f"🔄 Converting to WAV: {input_path}")
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.PROCESSING_TIMEOUT
            )
            
            if result.returncode != 0:
                raise AudioProcessingError(
                    "Failed to convert to WAV",
                    audio_path=input_path
                )
            
            logger.info(f"✅ Converted to WAV: {output_path}")
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(
                f"WAV conversion failed: {str(e)}",
                audio_path=input_path
            )
    
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg and FFprobe are installed"""
        try:
            # Check FFmpeg
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            # Check FFprobe
            subprocess.run(
                ['ffprobe', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
        except FileNotFoundError:
            raise AudioProcessingError(
                "FFmpeg not found. Please install FFmpeg.",
                details={
                    "install_hint": "Install FFmpeg: https://ffmpeg.org/download.html"
                }
            )
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to verify FFmpeg installation: {e}"
            )
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


# Singleton instance
_audio_processor_instance = None


def get_audio_processor() -> AudioProcessor:
    """
    Get AudioProcessor singleton instance
    
    Returns:
        AudioProcessor instance
    """
    global _audio_processor_instance
    if _audio_processor_instance is None:
        _audio_processor_instance = AudioProcessor()
    return _audio_processor_instance
