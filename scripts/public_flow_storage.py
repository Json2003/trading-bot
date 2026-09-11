"""GCS clients for public-flow archives; credentials stay in memory."""
import json
import os


def storage_client(*, read_only=False):
    from google.cloud import storage
    from google.oauth2 import service_account

    info = os.environ.get("PUBLIC_FLOW_GCS_CREDENTIALS")
    if not info:
        return storage.Client()  # Application default credentials on a persistent host.
    scope = "read_only" if read_only else "read_write"
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(info),
        scopes=[f"https://www.googleapis.com/auth/devstorage.{scope}"],
    )
    return storage.Client(project=credentials.project_id, credentials=credentials)
