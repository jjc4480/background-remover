import json
import boto3
import os
from io import BytesIO
from PIL import Image
import logging

# Disable numba caching for Lambda environment
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '1'
# Set rembg model path to /tmp (writable in Lambda)
os.environ['U2NET_HOME'] = '/tmp'
os.environ['HOME'] = '/tmp'

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

            # Remove background
            output_image = bg_remover.remove_background(input_image)

            # Save processed image to BytesIO
            output_buffer = BytesIO()

            # Always save as PNG to preserve transparency
            output_format = 'PNG'
            output_image.save(output_buffer, format=output_format)
            output_buffer.seek(0)

            # Upload processed image to S3
            # Extract path components: static/competition/applicant/[entry_no]/filename
            path_parts = key.split('/')
            filename = os.path.basename(key)
            name_without_ext = os.path.splitext(filename)[0]
            file_ext = os.path.splitext(filename)[1]

            # Ensure we have entry_no from path (static/competition/applicant/[entry_no]/filename)
            if len(path_parts) < 5:  # Need at least static/competition/applicant/entry_no/filename
                logger.error(f"Invalid path structure: {key}. Expected: static/competition/applicant/[entry_no]/filename")
                continue

            entry_no = path_parts[3]  # Extract entry_no from path
            directory = '/'.join(path_parts[:-1])  # Get directory path without filename

            # Save background removed image with _bg_removed suffix
            name_without_ext = os.path.splitext(filename)[0]
            output_key = f"{directory}/{name_without_ext}_bg_removed.png"
            s3_client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=output_buffer.getvalue(),
                ContentType='image/png'
            )
            logger.info(f"Background removed image saved: {output_key}")

            # Original file remains as is (no changes needed)
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