# ASR Worker - Orator AI

Python worker service for Automatic Speech Recognition (ASR) with GPU acceleration.

## Overview

This worker polls messages from AWS SQS ASR queue, processes audio files using GPU-accelerated speech-to-text (Whisper) and speaker diarization (pyannote.audio), achieving **11x faster processing** compared to CPU.

### Performance

- **GPU Processing**: 28-minute video processed in ~3 minutes (11x faster than CPU)
  - ASR: 17.91x realtime on NVIDIA GPU
  - Diarization: 20.52x realtime on NVIDIA GPU
- **CPU Processing**: 28-minute video takes ~34 minutes

## Features

- 🚀 **GPU Acceleration** - 11x faster processing with CUDA support
- 🎤 **Automatic Speech Recognition** - Whisper ASR with faster-whisper backend
- 👥 **Speaker Diarization** - GPU-accelerated pyannote.audio 3.1
- ☁️ **AWS Integration** - S3 for audio storage, SQS for job queue
- 🔄 **Async Processing** - Background job processing with retry logic
- 📊 **Webhook Callback** - Send results back to Node API

## Architecture

```
[AWS SQS ASR Queue]
        ↓
  [ASR Worker] (This service)
        ↓
1. Poll message from queue
2. Download audio from S3
3. Perform ASR (Speech-to-Text)
4. Perform Speaker Diarization
5. Send results to Node API webhook
6. Delete message from queue
```

## Tech Stack

- **Python 3.12+**
- **AWS SDK (boto3)** - S3, SQS
- **GPU Computing:**
  - PyTorch 2.5.1+cu121 (CUDA 12.1)
  - NVIDIA cuDNN 9.1.0
  - CUDA Toolkit 12.1+
- **Speech Recognition:**
  - faster-whisper 1.2.1 (GPU-accelerated Whisper)
  - ctranslate2 backend for efficient inference
- **Speaker Diarization:**
  - pyannote.audio 3.1.1 (GPU-enabled)
  - HuggingFace token required

## Project Structure

```
py-asr-worker/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Environment configuration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sqs_service.py      # SQS polling and message handling
│   │   ├── s3_service.py       # S3 download/upload
│   │   ├── asr_service.py      # Speech-to-text processing
│   │   ├── diarization_service.py  # Speaker separation
│   │   └── webhook_service.py  # API callback
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging setup
│       └── helpers.py          # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_asr_service.py
│   └── test_diarization_service.py
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- AWS account with S3 and SQS access
- FFmpeg (for audio processing)
- **For GPU acceleration (recommended):**
  - NVIDIA GPU with 6GB+ VRAM (e.g., GTX 1660 Ti or better)
  - CUDA Toolkit 12.1 or higher
  - cuDNN 9.1.0 or higher

### Setup

1. **Clone the repository** (if not already)

   ```bash
   cd py-asr-worker
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **For GPU support:**

   ```bash
   # Install PyTorch with CUDA 12.1
   pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

   # Install cuDNN
   pip install nvidia-cudnn-cu12
   ```

4. **Setup environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Install FFmpeg** (required for audio processing)

   ```bash
   # Windows (using chocolatey)
   choco install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Mac
   brew install ffmpeg
   ```

## Configuration

Create `.env` file with the following variables:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-southeast-1
AWS_S3_BUCKET=amzn-s3-oratorai
AWS_SQS_ASR_QUEUE_URL=https://sqs.ap-southeast-1.amazonaws.com/.../oratorai-asr-queue

# Node API Configuration
NODE_API_URL=http://localhost:8080
WEBHOOK_SECRET=your_webhook_secret

# ASR Configuration
ASR_ENGINE=whisper
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_LANGUAGE=vi # Vietnamese
WHISPER_DEVICE=cuda # Use 'cuda' for GPU (17x faster), 'cpu' for CPU
WHISPER_COMPUTE_TYPE=float16 # Use 'float16' for GPU, 'int8' for CPU

# Diarization Configuration
DIARIZATION_ENABLED=true
HUGGINGFACE_TOKEN=your_hf_token_here  # Get from https://huggingface.co/settings/tokens
MIN_SPEAKERS=1
MAX_SPEAKERS=5

# Worker Configuration
POLL_INTERVAL=5  # seconds
MAX_WORKERS=3
LOG_LEVEL=INFO
```

## Usage

### Run the worker

```bash
python src/main.py
```

### Run with Docker (optional)

```bash
docker build -t asr-worker .
docker run --env-file .env asr-worker
```

## Development

### Run tests

```bash
pytest tests/
```

### Run with hot reload

```bash
watchmedo auto-restart --directory=./src --pattern=*.py --recursive -- python src/main.py
```

## Message Format

### Input (from SQS Queue)

```json
{
  "jobId": 123,
  "presentationId": 456,
  "audioUrl": "https://s3.amazonaws.com/bucket/presentations/456/audio.mp3",
  "timestamp": "2026-01-22T10:30:00Z"
}
```

### Output (to Node API Webhook)

```json
{
  "jobId": 123,
  "presentationId": 456,
  "status": "success",
  "transcript": "Full transcription text...",
  "segments": [
    {
      "text": "Hello everyone",
      "startTime": 0.5,
      "endTime": 2.3,
      "speakerLabel": "SPEAKER_00",
      "confidence": 0.95
    }
  ],
  "diarization": [
    {
      "aiSpeakerLabel": "SPEAKER_00",
      "totalDuration": 120.5,
      "segmentCount": 15,
      "confidence": 0.92
    },
    {
      "aiSpeakerLabel": "SPEAKER_01",
      "totalDuration": 85.3,
      "segmentCount": 12,
      "confidence": 0.88
    }
  ],
  "metadata": {
    "language": "vi",
    "duration": 205.8,
    "audioFormat": "mp3",
    "processingTime": 45.2
  }
}
```

## Error Handling

- Automatic retry on failure (max 3 attempts)
- Failed jobs sent to DLQ (Dead Letter Queue)
- Error logging to CloudWatch
- Webhook notification on failure

## Monitoring

- CloudWatch Logs for application logs
- CloudWatch Metrics for processing metrics
- SQS metrics (messages in queue, processing time)

## Contributing

1. Create feature branch
2. Write tests
3. Submit pull request

## License

MIT

## Contact

Orator AI Team
