"""Real GCS SDK construction checks with ephemeral, unprivileged test keys."""

import json
import os
import socket
import subprocess
import unittest
from unittest.mock import patch

try:
    from google.cloud import storage
except ImportError:
    storage = None

from scripts.collect_trade_flow_window import GCSArchive


@unittest.skipIf(storage is None, "optional GCS SDK not installed; required in observer CI")
class TradeFlowSDKTests(unittest.TestCase):
    def test_real_sdk_constructs_in_memory_without_network(self):
        # The generated key has no account or permissions and is never persisted.
        pem = subprocess.run(
            ["openssl", "genpkey", "-algorithm", "RSA",
             "-pkeyopt", "rsa_keygen_bits:2048"],
            capture_output=True, text=True, check=True,
        ).stdout
        info = {
            "type": "service_account",
            "project_id": "offline-auth-test",
            "private_key_id": "synthetic-test-key",
            "private_key": pem,
            "client_email": "offline@offline-auth-test.iam.gserviceaccount.com",
            "client_id": "123456789",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with patch.dict(os.environ, {"GCP_SERVICE_ACCOUNT_KEY": json.dumps(info)}, clear=True):
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                with patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
                    archive = GCSArchive("offline-bucket", "/offline-prefix/")
        self.assertIsInstance(archive.client, storage.Client)
        self.assertEqual(archive.client.project, "offline-auth-test")
        self.assertEqual(archive.bucket.name, "offline-bucket")
        self.assertEqual(archive.prefix, "offline-prefix")

    def test_real_sdk_rejects_invalid_key_without_leaking_it(self):
        import traceback

        info = {
            "type": "service_account",
            "project_id": "offline-auth-test",
            "private_key": "synthetic-secret-not-a-pem",
            "client_email": "offline@offline-auth-test.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with patch.dict(os.environ, {"GCP_SERVICE_ACCOUNT_KEY": json.dumps(info)}, clear=True):
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                try:
                    GCSArchive("offline-bucket", "prefix")
                except ValueError:
                    rendered = traceback.format_exc()
                    self.assertNotIn("synthetic-secret-not-a-pem", rendered)
                    self.assertIn("not a valid service-account key", rendered)
                else:
                    self.fail("invalid PEM accepted")


if __name__ == "__main__":
    unittest.main()
