# ASR Worker - Orator AI

Python worker service for Automatic Speech Recognition (ASR) processing.

## Overview

This worker polls messages from AWS SQS ASR queue, processes audio files using speech-to-text technology, performs speaker diarization, and sends results back to Node API via webhook.

## Features

- 🎤 **Automatic Speech Recognition** - Convert audio to text
- 👥 **Speaker Diarization** - Identify and separate different speakers
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

- **Python 3.10+**
- **AWS SDK (boto3)** - S3, SQS
- **Speech Recognition:**
  - OpenAI Whisper (default)
  - Google Speech API (alternative)
  - Azure Speech Services (alternative)
- **Speaker Diarization:**
  - pyannote.audio
  - speechbrain (alternative)

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

- Python 3.10 or higher
- pip (Python package manager)
- AWS account with S3 and SQS access
- FFmpeg (for audio processing)

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
ASR_ENGINE=whisper  # Options: whisper, google, azure
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_LANGUAGE=vi # Vietnamese

# Diarization Configuration
DIARIZATION_ENABLED=true
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
