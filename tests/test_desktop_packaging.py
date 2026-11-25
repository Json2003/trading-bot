import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from scripts import package_desktop


@pytest.mark.parametrize("clean", [True, False])
def test_pyinstaller_builds_existing_spec(tmp_path, clean):
    spec = tmp_path / "Roman Bot.spec"
    spec.touch()
    work = tmp_path / "build files"
    dist = tmp_path / "dist files"

    with patch.object(package_desktop.subprocess, "run") as run:
        package_desktop.run_pyinstaller(spec, work, dist, clean=clean)

    expected = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        f"--workpath={work}",
        f"--distpath={dist}",
    ]
    if clean:
        expected.append("--clean")
    run.assert_called_once_with([*expected, str(spec)], check=True)


def test_checked_in_spec_preserves_asset_paths_in_portable_zip(tmp_path, monkeypatch):
    pytest.importorskip("PyInstaller")
    checkout = tmp_path / "checkout"
    source = checkout / "app sources"
    template = source / "templates" / "nested" / "test.html"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"<h1>RomanBot packaging test</h1>")
    spec = source / package_desktop.DEFAULT_SPEC.name
    shutil.copyfile(package_desktop.DEFAULT_SPEC, spec)
    shutil.copytree(package_desktop.DEFAULT_SPEC.parent / "pyinstaller_hooks", source / "pyinstaller_hooks")
    (checkout / "requests.py").write_text("raise RuntimeError('offline stub')\n", encoding="utf-8")
    (source / "roman_bot.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "import requests\n"
        "Path(sys.executable).with_name('requests-version.txt').write_text(requests.__version__)\n",
        encoding="utf-8",
    )
    release = tmp_path / "release files"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_desktop.py",
            "--spec",
            str(spec),
            "--build-dir",
            str(tmp_path / "build files"),
            "--dist-dir",
            str(tmp_path / "dist files"),
            "--release-dir",
            str(release),
            "--zip",
        ],
    )
    package_desktop.main()

    executable = "RomanBot.exe" if os.name == "nt" else "RomanBot"
    with ZipFile(release / "RomanBot.zip") as archive:
        assert executable in archive.namelist()
        assert archive.read("_internal/templates/nested/test.html") == template.read_bytes()

    bundle = release / "RomanBot"
    subprocess.run([str(bundle / executable)], check=True, timeout=30)
    assert (bundle / "requests-version.txt").read_text(encoding="utf-8") == version("requests")


def test_packaging_dependencies_write_and_read_parquet(tmp_path):
    pytest.importorskip("PyInstaller")
    import duckdb
    import pandas as pd

    path = tmp_path / "sample.parquet"
    pd.DataFrame({"quantity": [1, 2]}).to_parquet(path, engine="pyarrow", index=False)

    with duckdb.connect(":memory:") as connection:
        assert connection.execute("SELECT sum(quantity) FROM read_parquet(?)", [str(path)]).fetchone() == (3,)


def test_launcher_creates_window_with_installed_pywebview(tmp_path, monkeypatch):
    pytest.importorskip("PyInstaller")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PYSTRAY_BACKEND", "dummy")
    from tradingbot_ibkr import roman_bot

    monkeypatch.setattr(roman_bot.webview, "windows", [])
    with (
        patch.object(roman_bot.threading, "Thread"),
        patch.object(roman_bot.time, "sleep"),
        patch.object(roman_bot.webview, "start") as start,
    ):
        roman_bot.main()

    start.assert_called_once_with()
    assert len(roman_bot.webview.windows) == 1
    assert roman_bot.webview.windows[0].original_url == "http://127.0.0.1:5001"
