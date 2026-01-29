#!/usr/bin/env python3
"""Purge SQS queue to remove old jobs"""
import boto3
import os

# Get credentials from environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
QUEUE_URL = os.getenv("AWS_SQS_ASR_QUEUE_URL")

# Create SQS client
sqs = boto3.client(
    'sqs',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

print(f"🗑️  Purging queue: {QUEUE_URL}")

try:
    response = sqs.purge_queue(QueueUrl=QUEUE_URL)
    print("✅ Queue purged successfully!")
    print("   All messages deleted from queue")
    print("   You can now upload a new video")
except Exception as e:
    print(f"❌ Error: {e}")
