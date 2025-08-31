"""Upload files to GCS using google-cloud-storage.

Usage example (local):
  # set GOOGLE_APPLICATION_CREDENTIALS env var to service account JSON
  python upload_to_gcs.py --bucket historicaltradedataromantradebot --local-dir tradingbot_ibkr/datafiles --dest-path data
"""
import argparse
from pathlib import Path
from google.cloud import storage
import os


def upload_dir(bucket_name, local_dir, dest_path=''):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_dir = Path(local_dir)
    files = list(local_dir.rglob('*'))
    uploaded = 0
    for f in files:
        if f.is_file():
            rel = f.relative_to(local_dir)
            blob_name = str(Path(dest_path) / rel).replace('\\', '/')
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(f))
            uploaded += 1
            print('Uploaded', blob_name)
    print('Uploaded', uploaded, 'files')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket', required=True)
    p.add_argument('--local-dir', required=True)
    p.add_argument('--dest-path', default='')
    args = p.parse_args()
    upload_dir(args.bucket, args.local_dir, args.dest_path)


if __name__ == '__main__':
    main()
