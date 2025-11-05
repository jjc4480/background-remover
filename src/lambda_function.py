import json
import boto3
import os
from io import BytesIO
from PIL import Image
import logging

# Disable numba caching for Lambda environment
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['HOME'] = '/tmp'

# MediaPipe configuration for Lambda (must be set BEFORE importing mediapipe)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TMPDIR'] = '/tmp'

# BiRefNet 모델을 Docker 이미지 /opt에서 /tmp로 복사 (최초 1회, Lambda container 재사용 시 skip)
import shutil
import sys

u2net_cache = '/tmp/.u2net'
model_src = '/opt/models/birefnet-general.onnx'
model_dst = f'{u2net_cache}/birefnet-general.onnx'

if not os.path.exists(model_dst):
    os.makedirs(u2net_cache, exist_ok=True)
    # Docker 이미지에 모델이 있으면 복사, 없으면 나중에 자동 다운로드됨
    if os.path.exists(model_src):
        try:
            shutil.copy2(model_src, model_dst)
            print(f"BiRefNet model copied: {model_src} -> {model_dst}")
        except Exception as e:
            print(f"Warning: Failed to copy model: {e}")
    else:
        print(f"Model not found at {model_src}, will download on first use")
os.environ['U2NET_HOME'] = u2net_cache

# MediaPipe 모델을 /opt에서 import하도록 sys.path 추가
if '/opt' not in sys.path:
    sys.path.insert(0, '/opt')

from background_remover import BackgroundRemover

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to remove background from images uploaded to S3
    """
    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']

            # Check if image is in static/competition/applicant path
            if not key.startswith('static/competition/applicant/'):
                logger.info(f"Skipping {key} - not in target path (static/competition/applicant/)")
                continue

            # Skip if already processed files (bg_removed suffix)
            filename = os.path.basename(key)
            if 'bg_removed' in filename:
                logger.info(f"Skipping already processed image: {key}")
                continue

            # Check if this file was already processed (look for bg_removed version)
            directory = '/'.join(key.split('/')[:-1])
            name_without_ext = os.path.splitext(filename)[0]
            bg_removed_key = f"{directory}/{name_without_ext}_bg_removed.png"

            # Check if bg_removed version exists (indicates already processed)
            try:
                s3_client.head_object(Bucket=bucket, Key=bg_removed_key)
                logger.info(f"Skipping - already processed (bg_removed exists): {key}")
                continue
            except:
                # bg_removed doesn't exist, proceed with processing
                pass

            logger.info(f"Processing image: {bucket}/{key}")

            # Download image from S3
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()

            # Open image
            input_image = Image.open(BytesIO(image_data))

            # Initialize background remover with BiRefNet model
            bg_remover = BackgroundRemover(model='birefnet-general')

            # Remove background (returns tuple: version1, version2)
            version1, version2 = bg_remover.remove_background(input_image, filename=filename)

            # Upload processed images to S3
            # Extract path components: static/competition/applicant/[entry_no]/filename
            path_parts = key.split('/')
            filename = os.path.basename(key)
            name_without_ext = os.path.splitext(filename)[0]

            # Ensure we have entry_no from path (static/competition/applicant/[entry_no]/filename)
            if len(path_parts) < 5:  # Need at least static/competition/applicant/entry_no/filename
                logger.error(f"Invalid path structure: {key}. Expected: static/competition/applicant/[entry_no]/filename")
                continue

            directory = '/'.join(path_parts[:-1])  # Get directory path without filename

            # Save version 1: _bg_removed.png (600x640 상반신)
            buffer1 = BytesIO()
            version1.save(buffer1, format='PNG')
            buffer1.seek(0)
            output_key1 = f"{directory}/{name_without_ext}_bg_removed.png"
            s3_client.put_object(
                Bucket=bucket,
                Key=output_key1,
                Body=buffer1.getvalue(),
                ContentType='image/png'
            )
            logger.info(f"Version 1 (bg_removed) saved: {output_key1}")

            # Save version 2: _bg_removed_for_award.png (시상용)
            buffer2 = BytesIO()
            version2.save(buffer2, format='PNG')
            buffer2.seek(0)
            output_key2 = f"{directory}/{name_without_ext}_bg_removed_for_award.png"
            s3_client.put_object(
                Bucket=bucket,
                Key=output_key2,
                Body=buffer2.getvalue(),
                ContentType='image/png'
            )
            logger.info(f"Version 2 (for_award) saved: {output_key2}")

            # Original file remains as is
            logger.info(f"Original file kept: {key}")

            logger.info(f"Processing completed for: {key}")

        return {
            'statusCode': 200,
            'body': json.dumps('Background removal completed successfully')
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }