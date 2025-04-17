import os
import boto3
import logging
from botocore.exceptions import ClientError
from typing import List, Optional

def get_s3_client():
    """
    Create and return an S3 client.
    
    Returns:
        boto3.client: Configured S3 client
    """
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

def upload_file_to_s3(file_path: str, bucket: str, object_name: Optional[str] = None) -> bool:
    """
    Upload a file to an S3 bucket.
    
    Args:
        file_path: Path to the file to upload
        bucket: Bucket name
        object_name: S3 object name (if None, uses file_path)
    
    Returns:
        bool: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    s3_client = get_s3_client()
    try:
        s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logging.error(f"Failed to upload file to S3: {e}")
        return False
    return True

def download_file_from_s3(bucket: str, object_name: str, file_path: str) -> bool:
    """
    Download a file from an S3 bucket.
    
    Args:
        bucket: Bucket name
        object_name: S3 object name
        file_path: Path to save the downloaded file
    
    Returns:
        bool: True if file was downloaded, else False
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    s3_client = get_s3_client()
    try:
        s3_client.download_file(bucket, object_name, file_path)
    except ClientError as e:
        logging.error(f"Failed to download file from S3: {e}")
        return False
    return True

def upload_directory_to_s3(directory: str, bucket: str, prefix: str = "") -> List[str]:
    """
    Upload an entire directory to S3.
    
    Args:
        directory: Local directory to upload
        bucket: S3 bucket name
        prefix: Prefix to add to S3 object keys
    
    Returns:
        List[str]: List of uploaded object keys
    """
    uploaded_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_path = os.path.join(prefix, relative_path).replace("\\", "/")
            
            if upload_file_to_s3(local_path, bucket, s3_path):
                uploaded_files.append(s3_path)
    
    return uploaded_files