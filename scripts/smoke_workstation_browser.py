import os
import re
import socket
import sys
import tempfile
import time
from pathlib import Path
from subprocess import DEVNULL, Popen
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "only-for-isolated-workstation-smoke-test"
ARTIFACTS = ROOT / "var" / "workstation-smoke"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def wait_for_operator(url, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Workstation operator exited before starting")
        try:
            with urlopen(url + "/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("Workstation operator did not become reachable")


def test_workstation(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(url + "/workstation")
        expect(page.get_by_text("Operator locked")).to_be_visible()
        expect(page.locator("#startButton")).to_be_disabled()
        page.locator("#accessToken").fill("wrong-test-token")
        page.get_by_role("button", name="Connect", exact=True).click()
        expect(page.locator("#notice")).to_contain_text("invalid operator token")
        page.locator("#accessToken").fill(TOKEN)
        page.get_by_role("button", name="Connect", exact=True).click()
        expect(page.get_by_text("Operator online")).to_be_visible()
        expect(page.locator("#access")).to_be_hidden()
        expect(page.locator("#equity")).to_have_text("$5,000.00")
        expect(page.locator("#chart")).to_have_attribute(
            "aria-label", "IBM · 78 five-minute bars · synthetic",
        )
        assert page.locator("#chart rect").count() >= 78
        page.screenshot(path=ARTIFACTS / "workstation-browser.png", full_page=True)
        page.locator("[data-symbol=JPM]").click()
        page.get_by_role("button", name="1H", exact=True).click()
        expect(page.locator("#chart")).to_have_attribute(
            "aria-label", "JPM · 12 five-minute bars · synthetic",
        )
        page.locator("#startButton").click()
        expect(page.locator("#engineStatus")).to_have_text("Running")
        expect(page.locator("#positionMetric")).not_to_have_text("0")
        page.locator("#pauseButton").click()
        expect(page.locator("#engineStatus")).to_have_text("Paused")
        page.get_by_role("button", name="IBKR Live", exact=True).click()
        expect(page.locator("#simControls")).to_be_hidden()
        expect(page.locator("#equity")).to_have_text("—")
        expect(page.locator("#chart")).to_be_hidden()
        page.locator("#checkTws").click()
        expect(page.locator("#notice")).to_contain_text("IBKR_TWS_LIVE_ACCOUNT")
        page.get_by_role("button", name="IBKR Paper", exact=True).click()
        page.locator("#checkTws").click()
        expect(page.locator("#notice")).to_contain_text("IBKR_TWS_PAPER_ACCOUNT")
        page.get_by_role("button", name="TWS connection", exact=True).click()
        expect(page.locator("#paperPort")).to_have_text("7497")
        expect(page.locator("#livePort")).to_have_text("7496")
        page.screenshot(path=ARTIFACTS / "tws-connection.png", full_page=True)
        page.get_by_role("button", name="Orders", exact=True).click()
        expect(page.locator("#ordersEmpty")).to_contain_text("not retrieved")
        page.get_by_role("button", name="Simulator", exact=True).click()
        page.get_by_role("button", name="Overview", exact=True).click()
        page.locator("#emergencyButton").click()
        expect(page.locator("#engineStatus")).to_have_text("Stop latched")
        expect(page.locator("#startButton")).to_be_disabled()
        page.get_by_role("button", name="Research Lab", exact=True).click()
        expect(page.locator("#labStartButton")).to_be_enabled()
        page.locator("#labDataset").select_option("sample_ohlcv.csv")
        page.locator("#labGenerations").fill("1")
        page.locator("#labAccounts").fill("4")
        page.locator("#labStartButton").click()
        state = page.locator("#labJobState")
        expect(state).to_have_text(re.compile(r"^(completed|failed|cancelled|interrupted)$"), timeout=25000)
        assert state.text_content() == "completed", page.locator("#labLeaderboardEmpty").text_content()
        expect(page.locator("#labLeaderboardBody > tr")).to_have_count(1)
        page.get_by_role("button", name="Overview", exact=True).click()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth === window.innerWidth")
        page.screenshot(path=ARTIFACTS / "workstation-mobile.png", full_page=True)
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as data_dir, socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
        port_probe.close()
        env = {
            **{key: value for key, value in os.environ.items() if not key.startswith("IBKR_TWS_")},
            "TRADING_OPERATOR_TOKEN": TOKEN,
            "TRADING_OPERATOR_HOST": "127.0.0.1",
            "TRADING_OPERATOR_PORT": str(port),
            "TRADING_OPERATOR_RUNTIME": "nyse-demo",
            "TRADING_OPERATOR_CYCLE_SECONDS": "0.3",
            "DISABLE_STUBS": "1",
            "TRADING_RESEARCH_ARTIFACT_ROOT": data_dir,
            "TRADING_RESEARCH_DATASET_ROOT": str(ROOT / "backtest" / "sample_data"),
        }
        with Popen(
            [sys.executable, "-m", "scripts.run_operator_api"],
            cwd=ROOT, env=env, stdout=DEVNULL,
        ) as process:
            try:
                url = f"http://127.0.0.1:{port}"
                wait_for_operator(url, process)
                test_workstation(url)
            finally:
                process.terminate()
                process.wait(timeout=10)
    print("Workstation browser smoke passed")
