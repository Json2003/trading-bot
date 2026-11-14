"""Offline authentication and workflow regression tests (standard library only)."""

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import types
import unittest
from unittest.mock import Mock, patch

from scripts.collect_trade_flow_window import GCSArchive, _initialize_window_manifest


class TradeFlowAuthTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        storage = types.ModuleType("google.cloud.storage")
        storage.Client = self.client
        cloud = types.ModuleType("google.cloud")
        cloud.storage = storage
        google = types.ModuleType("google")
        google.cloud = cloud
        self.modules = patch.dict(sys.modules, {
            "google": google, "google.cloud": cloud, "google.cloud.storage": storage,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_existing_json_is_used_in_memory(self):
        info = {"type": "service_account", "project_id": "offline-test",
                "private_key": "synthetic-marker"}
        os.environ["GCP_SERVICE_ACCOUNT_KEY"] = json.dumps(info)
        with patch("builtins.open", side_effect=AssertionError("no credential files")):
            archive = GCSArchive("existing-bucket", "/existing/prefix/")
        self.client.from_service_account_info.assert_called_once_with(info)
        self.client.assert_not_called()
        archive.client.bucket.assert_called_once_with("existing-bucket")
        self.assertEqual(archive.prefix, "existing/prefix")

    def test_absent_environment_keeps_adc(self):
        archive = GCSArchive("existing-bucket", "prefix")
        self.client.assert_called_once_with()
        self.client.from_service_account_info.assert_not_called()
        self.assertIs(archive.client, self.client.return_value)

    def test_invalid_input_fails_closed_without_secret_in_traceback(self):
        for value in ["", "synthetic-secret", "null", "[]", "{}",
                      '{"type":"external_account"}', '{"type":"authorized_user"}']:
            with self.subTest(value=value):
                os.environ["GCP_SERVICE_ACCOUNT_KEY"] = value
                try:
                    GCSArchive("bucket", "prefix")
                except ValueError:
                    rendered = traceback.format_exc()
                    self.assertNotIn("synthetic-secret", rendered)
                    self.assertIn("not a valid service-account key", rendered)
                else:
                    self.fail("invalid credential input accepted")
        self.client.assert_not_called()
        self.client.from_service_account_info.assert_not_called()

    def test_sdk_parse_errors_are_sanitized_and_do_not_use_adc(self):
        os.environ["GCP_SERVICE_ACCOUNT_KEY"] = '{"type":"service_account"}'
        for error in [ValueError, TypeError, KeyError]:
            with self.subTest(error=error):
                self.client.from_service_account_info.side_effect = error("synthetic-secret")
                try:
                    GCSArchive("bucket", "prefix")
                except ValueError:
                    self.assertNotIn("synthetic-secret", traceback.format_exc())
                else:
                    self.fail("SDK rejection did not fail closed")
        self.client.assert_not_called()

    def test_workflow_preserves_owner_restriction_and_gates(self):
        workflow = Path(".github/workflows/public-trade-flow-collect.yml").read_text()
        actions = [line.strip().split("uses: ", 1)[1]
                   for line in workflow.splitlines() if "uses: " in line]
        self.assertEqual(actions, ["actions/checkout@v4", "actions/setup-python@v5",
                                   "actions/upload-artifact@v4"])
        self.assertNotIn("google-github-actions", workflow)
        collect = workflow.split("- name: Collect and durably checkpoint one segment", 1)[1]
        self.assertIn("GCP_SERVICE_ACCOUNT_KEY: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}", collect)
        self.assertIn("contents: read", workflow)
        self.assertIn("if: github.ref == 'refs/heads/main'", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("--fail-on-gap", workflow)
        self.assertNotIn("id-token:", workflow)

    def test_frozen_window_and_no_trading_flags_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest, created = _initialize_window_manifest(path, "test", ["BTCUSDT"], 100000, 90)
            self.assertTrue(created)
            for flag in ["orders_allowed", "paper_orders_placed", "live_orders_placed",
                         "leverage_enabled", "active_profile_changed", "promotion_allowed"]:
                self.assertIs(manifest[flag], False)
            self.assertIs(manifest["research_only"], True)
            with self.assertRaises(ValueError):
                _initialize_window_manifest(path, "test", ["ETHUSDT"], 100000, 90)


if __name__ == "__main__":
    unittest.main()
