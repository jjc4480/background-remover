import json
import boto3
import os
import urllib.parse
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    if os.path.exists(model_src):
        try:
            shutil.copy2(model_src, model_dst)
            logger.info(f"BiRefNet model copied: {model_src} -> {model_dst}")
        except Exception as e:
            logger.warning(f"Failed to copy model: {e}")
    else:
        logger.warning(f"Model not found at {model_src}, will download on first use")
os.environ['U2NET_HOME'] = u2net_cache

# MediaPipe 모델을 /opt에서 import하도록 sys.path 추가
if '/opt' not in sys.path:
    sys.path.insert(0, '/opt')

from background_remover import BackgroundRemover
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')
bg_remover = BackgroundRemover(model='birefnet-general')


def _upload_png(image, bucket, key):
    """PIL Image를 PNG로 S3에 업로드"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType='image/png')
    logger.info(f"Uploaded: {key}")


def lambda_handler(event, context):
    """
    Lambda function to remove background from images uploaded to S3
    """
    results = []
    errors = []

    for record in event['Records']:
        try:
            bucket = record['s3']['bucket']['name']
            key = urllib.parse.unquote_plus(record['s3']['object']['key'])

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

            # Ensure we have entry_no from path (static/competition/applicant/[entry_no]/filename)
            path_parts = key.split('/')
            if len(path_parts) < 5:
                logger.error(f"Invalid path structure: {key}. Expected: static/competition/applicant/[entry_no]/filename")
                continue

            # Check if bg_removed version exists (indicates already processed)
            try:
                s3_client.head_object(Bucket=bucket, Key=bg_removed_key)
                logger.info(f"Skipping - already processed (bg_removed exists): {key}")
                continue
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
                # 404 = bg_removed doesn't exist, proceed with processing

            logger.info(f"Processing image: {bucket}/{key}")

            # Download image from S3
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()

            # Open image
            input_image = Image.open(BytesIO(image_data))

            # Remove background (returns tuple: version1, version2)
            version1, version2 = bg_remover.remove_background(input_image, filename=filename)

            # Upload processed images to S3
            _upload_png(version1, bucket, f"{directory}/{name_without_ext}_bg_removed.png")
            _upload_png(version2, bucket, f"{directory}/{name_without_ext}_bg_removed_for_award.png")

            logger.info(f"Processing completed for: {key}")
            results.append(key)

        except Exception as e:
            error_key = record.get('s3', {}).get('object', {}).get('key', 'unknown')
            logger.error(f"Error processing {error_key}: {str(e)}", exc_info=True)
            errors.append(f"{error_key}: {str(e)}")

    if errors:
        return {
            'statusCode': 207,
            'body': json.dumps({'processed': results, 'errors': errors})
        }
    return {
        'statusCode': 200,
        'body': json.dumps('Background removal completed successfully')
    }
