#!/usr/bin/env python3
"""Build a desktop installer for the Roman Bot launcher.

This script wraps PyInstaller (and optionally Inno Setup on Windows) so the
packaging process can be reproduced with a single command.

Usage examples:
    python scripts/package_desktop.py
    python scripts/package_desktop.py --spec tradingbot_ibkr/roman_bot.spec --release-dir dist/desktop
    python scripts/package_desktop.py --windows-installer
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "tradingbot_ibkr" / "roman_bot.spec"
DEFAULT_RELEASE_DIR = REPO_ROOT / "release" / "desktop"
INNO_SETUP_SCRIPT = REPO_ROOT / "tradingbot_ibkr" / "romanbot_installer.iss"


def run_pyinstaller(spec_path: Path, build_dir: Path, dist_dir: Path, clean: bool = True) -> None:
    """Invoke PyInstaller against the provided spec file."""
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        f"--workpath={build_dir}",
        f"--distpath={dist_dir}",
    ]
    if clean:
        cmd.append("--clean")
    cmd.append(str(spec_path))
    print(f"[PyInstaller] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_inno_setup(output_dir: Path, release_dir: Path) -> None:
    """Compile the Inno Setup script if ISCC is available."""
    if platform.system() != "Windows":
        print("[Inno Setup] Skipped (only supported on Windows).")
        return
    iscc = shutil.which("ISCC")
    if not iscc:
        print("[Inno Setup] ISCC.exe not found in PATH. Skipping installer build.")
        return
    if not INNO_SETUP_SCRIPT.exists():
        print(f"[Inno Setup] Script not found: {INNO_SETUP_SCRIPT}. Skipping installer build.")
        return

    env = os.environ.copy()
    env["OUTPUT_DIR"] = str(output_dir)
    env["RELEASE_DIR"] = str(release_dir)

    cmd = [iscc, "/Q", str(INNO_SETUP_SCRIPT)]
    print(f"[Inno Setup] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def copy_release(bin_dir: Path, release_dir: Path) -> Path:
    """Copy the PyInstaller dist directory to the release folder."""
    if not bin_dir.exists():
        raise FileNotFoundError(f"PyInstaller output not found: {bin_dir}")

    release_dir.mkdir(parents=True, exist_ok=True)
    target = release_dir / bin_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(bin_dir, target)
    print(f"[Release] Copied build to {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Roman Bot as a desktop application.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Path to PyInstaller spec file.")
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "desktop", help="PyInstaller build directory.")
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist" / "desktop", help="PyInstaller dist directory.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR, help="Directory to copy final artefacts.")
    parser.add_argument("--no-clean", action="store_true", help="Do not pass --clean to PyInstaller.")
    parser.add_argument("--windows-installer", action="store_true", help="Attempt to build an Inno Setup installer (Windows only).")
    parser.add_argument("--skip-release-copy", action="store_true", help="Do not copy dist output into release directory.")
    parser.add_argument("--zip", dest="zip_archive", action="store_true", help="Create a ZIP archive of the release directory.")
    return parser.parse_args()


def create_zip_archive(source_dir: Path, release_dir: Path) -> Path:
    """Create a ZIP archive of the built application for easy distribution."""
    archive_name = release_dir / f"{source_dir.name}"
    if archive_name.with_suffix(".zip").exists():
        archive_name.with_suffix(".zip").unlink()
    zip_path = shutil.make_archive(str(archive_name), "zip", root_dir=source_dir)
    print(f"[Archive] Created ZIP at {zip_path}")
    return Path(zip_path)


def main() -> None:
    args = parse_args()

    try:
        run_pyinstaller(args.spec, args.build_dir, args.dist_dir, clean=not args.no_clean)
        bin_dir = args.dist_dir / "RomanBot"
        release_target = None
        if not args.skip_release_copy:
            release_target = copy_release(bin_dir, args.release_dir)
        else:
            release_target = bin_dir
        archive_path = None
        if args.zip_archive:
            archive_path = create_zip_archive(release_target, args.release_dir)
        if args.windows_installer:
            run_inno_setup(release_target, args.release_dir)
        print("Packaging completed successfully.")
        if archive_path:
            print(f"ZIP archive available at: {archive_path}")
    except subprocess.CalledProcessError as exc:
        print(f"Packaging failed: {exc}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"Packaging failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
