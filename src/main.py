"""
ASR Worker - Main Entry Point

This worker polls messages from AWS SQS, processes audio files,
performs speech-to-text and speaker diarization, then sends results
back to Node API via webhook.
"""

import sys
import signal
import time
from typing import Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ASRWorker:
    """Main ASR Worker class"""
    
    def __init__(self):
        self.running = False
        self.worker_name = f"ASRWorker-{settings.WORKER_ID}"
        
    def start(self):
        """Start the worker"""
        logger.info(f"🚀 Starting {self.worker_name}")
        logger.info(f"📊 Configuration:")
        logger.info(f"   - ASR Engine: {settings.ASR_ENGINE}")
        logger.info(f"   - Whisper Model: {settings.WHISPER_MODEL}")
        logger.info(f"   - Diarization: {settings.DIARIZATION_ENABLED}")
        logger.info(f"   - Poll Interval: {settings.POLL_INTERVAL}s")
        logger.info(f"   - Max Workers: {settings.MAX_WORKERS}")
        
        self.running = True
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("⚠️ Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)
    
    def _run_loop(self):
        """Main worker loop"""
        logger.info("🔄 Worker started, polling for messages...")
        
        while self.running:
            try:
                # TODO: Poll SQS for messages
                logger.debug(f"📥 Polling SQS queue...")
                
                # Placeholder: will implement SQS polling
                time.sleep(settings.POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Error in worker loop: {e}", exc_info=True)
                time.sleep(settings.POLL_INTERVAL)
    
    def stop(self):
        """Stop the worker gracefully"""
        logger.info(f"🛑 Stopping {self.worker_name}...")
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
