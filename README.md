# ASR Worker — OratorAI

Python worker service xử lý **Automatic Speech Recognition (ASR)** với GPU acceleration, dành riêng cho hệ thống OratorAI.

## Tổng quan

Worker poll message từ **AWS SQS ASR Queue**, download audio từ S3, thực hiện speech-to-text (Whisper GPU) + speaker diarization (pyannote.audio), rồi gửi kết quả về Node API qua webhook.

### Hiệu suất

- **GPU (GTX 1660 Ti / RTX 3060)**: video 28 phút xử lý ~3 phút (11x nhanh hơn CPU)
  - ASR: 17.9x realtime
  - Diarization: 20.5x realtime
- **CPU**: video 28 phút xử lý ~34 phút

---

## Pipeline xử lý

```
[AWS SQS ASR Queue]
        ↓
1. Poll message
2. Download audio từ S3
3. Validate & normalize audio (FFmpeg)
4. ASR — Speech-to-Text (faster-whisper)
5. Speaker Diarization (pyannote.audio)
6. Merge transcript + diarization
7. Gửi kết quả → Node API webhook
8. Xóa message khỏi queue
```

---

## Tech Stack

| Thành phần | Thư viện / Version |
|---|---|
| Runtime | Python 3.12+ |
| AWS SDK | boto3 |
| GPU | PyTorch 2.5.1+cu121, CUDA 12.1, cuDNN 9.x |
| ASR | faster-whisper 1.2.1 (ctranslate2 backend) |
| Diarization | pyannote.audio 3.1.1 |
| Audio | FFmpeg, pydub, librosa, soundfile |

---

## Cấu trúc project

```
py-asr-worker/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point — ASRWorker class
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic settings từ .env
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sqs_service.py         # Poll & delete SQS messages
│   │   ├── s3_service.py          # Download audio từ S3
│   │   ├── audio_processor.py     # Validate & normalize audio
│   │   ├── asr_service.py         # Speech-to-text (Whisper)
│   │   ├── diarization_service.py # Speaker separation (pyannote)
│   │   ├── webhook_service.py     # Gửi kết quả về Node API
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Colorlog setup
│       ├── exceptions.py          # Custom exception classes
│       ├── helpers.py             # Utility functions
│       └── validators.py          # Audio validation helpers
├── tests/
│   └── __init__.py
├── logs/                          # Log files (gitignored)
├── Dockerfile                     # Single-stage GPU image (CUDA 12.1)
├── docker-compose.yml             # Compose với GPU + named volumes
├── requirements.txt
├── .env.example                   # Template biến môi trường
└── purge_queue.py                 # Script xóa tất cả SQS messages
```

---

## Cài đặt & Chạy

### Cách 1: Docker (Khuyến nghị)

**Yêu cầu:**
- Docker Desktop (Windows) với WSL2 backend
- NVIDIA GPU driver ≥ 530
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Windows: tích hợp sẵn trong Docker Desktop)

```bash
# 1. Copy và điền thông tin vào .env
cp .env.example .env

# 2. Build image (lần đầu ~15-20 phút — download PyTorch CUDA ~2GB)
docker compose build

# 3. Chạy worker
docker compose up -d

# 4. Xem logs
docker compose logs -f asr-worker

# 5. Kiểm tra GPU hoạt động
docker exec py-asr-worker python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

> **Lưu ý:** Lần đầu khởi động sẽ tự động download Whisper model `large-v2` (~3GB) và pyannote diarization model (~1GB) vào named volumes. Mất 5–15 phút tùy tốc độ mạng.

---

### Cách 2: Chạy trực tiếp (Windows — môi trường dev)

**Yêu cầu:**
- Python 3.12
- CUDA 12.1 + cuDNN 9.x đã cài
- FFmpeg trong PATH

```bash
# 1. Tạo virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Cài PyTorch CUDA 12.1 (PHẢI cài TRƯỚC requirements.txt)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các dependency còn lại
pip install -r requirements.txt

# 4. Setup .env
cp .env.example .env
# Điền AWS keys, HuggingFace token, webhook secret vào .env

# 5. Chạy worker
python src/main.py
```

---

## Cấu hình quan trọng (`.env`)

```env
# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-southeast-1
AWS_S3_BUCKET=amzn-s3-oratorai
AWS_SQS_ASR_QUEUE_URL=https://sqs.ap-southeast-1.amazonaws.com/.../oratorai-asr-queue
AWS_SQS_ANALYST_QUEUE_URL=https://sqs.ap-southeast-1.amazonaws.com/.../oratorai-analysis-queue

# Webhook về Node API
NODE_API_URL=https://your-node-api.app     # KHÔNG có trailing slash
WEBHOOK_ENDPOINT=/api/v1/webhooks/asr-complete
WEBHOOK_SECRET=your_webhook_secret

# ASR — Whisper
WHISPER_MODEL=large-v2
WHISPER_DEVICE=cuda
# Chọn compute type phù hợp với VRAM:
#   int8       → ~1.5GB VRAM — dùng cho GPU 6GB (GTX 1660 Ti)
#   float16    → ~3.0GB VRAM — dùng cho GPU ≥8GB (RTX 3060+)
WHISPER_COMPUTE_TYPE=int8
BEAM_SIZE=5                                 # Không tăng lên >5 để tránh OOM trên 6GB VRAM

# Diarization
DIARIZATION_ENABLED=true
HUGGINGFACE_TOKEN=hf_...                    # Lấy tại: huggingface.co/settings/tokens
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
```

---

## Format dữ liệu

### Input — SQS Message

```json
{
  "jobId": 123,
  "presentationId": 456,
  "audioUrl": "s3://amzn-s3-oratorai/presentations/456/audio.mp3",
  "metadata": {}
}
```

### Output — Webhook về Node API

```json
{
  "jobId": 123,
  "presentationId": 456,
  "status": "completed",
  "transcript": [
    {
      "id": 1,
      "text": "Xin chào mọi người",
      "start": 0.5,
      "end": 2.3,
      "speakerLabel": "SPEAKER_00",
      "confidence": 0.95
    }
  ],
  "speakers": [
    {
      "speakerLabel": "SPEAKER_00",
      "totalDuration": 120.5,
      "segmentCount": 15,
      "confidence": 0.92,
      "percentage": 65.0
    }
  ],
  "metadata": {
    "audioDuration": 205.8,
    "whisperModel": "large-v2",
    "whisperLanguage": "vi",
    "diarizationEnabled": true,
    "processingTime": 45.2
  }
}
```

---

## Error Handling

- **Retry tự động** — tối đa 3 lần (cấu hình `MAX_RETRIES`)
- **Message không bị xóa** nếu lỗi → SQS tự requeue sau `VISIBILITY_TIMEOUT` giây
- **Webhook thất bại** — gửi payload `status: "failed"` về Node API
- **Log file** — lưu tại `logs/asr-worker.log` (rotate tự động)

---

## Kiểm tra GPU

```bash
# Kiểm tra CUDA có khả dụng không
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Kiểm tra VRAM còn trống
python -c "import torch; print(f'Free VRAM: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB')"
```

---

## Lệnh Docker hữu ích

```bash
# Xem trạng thái container
docker compose ps

# Restart worker
docker compose restart asr-worker

# Xem log realtime
docker compose logs -f asr-worker

# Xóa tất cả messages trong queue (dev)
python purge_queue.py

# Dừng worker
docker compose down
```
