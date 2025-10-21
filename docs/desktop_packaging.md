# Desktop Packaging Guide

Use this guide to build a desktop-ready version of the Roman Bot launcher and produce an installer for distribution.

## 1. Prerequisites

- Python 3.10 or newer.
- Virtual environment activated for the repo (recommended).
- PyInstaller (install via `pip install -r requirements-packaging.txt`).
- On Windows (optional, for MSI/EXE installer):
  - [Inno Setup](https://jrsoftware.org/isinfo.php) installed and `ISCC.exe` available on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r tradingbot_ibkr/requirements.txt
pip install -r requirements-packaging.txt
```

## 2. Build the desktop bundle

The automation script wraps PyInstaller, collects artefacts, and (optionally) triggers Inno Setup.

```bash
python scripts/package_desktop.py --zip
```

Outputs:

- `dist/desktop/RomanBot/` – PyInstaller build.
- `release/desktop/RomanBot/` – copy ready for distribution.
- `release/desktop/RomanBot.zip` – zipped package when using `--zip` flag.
- `release/desktop/` will contain the Inno installer when built on Windows with `--windows-installer`.

### Useful flags

- `--spec` – use a custom `.spec` file (defaults to `tradingbot_ibkr/roman_bot.spec`).
- `--windows-installer` – attempt to compile `romanbot_installer.iss` via Inno Setup (Windows only).
- `--no-clean` – reuse existing PyInstaller cache for faster rebuilds.
- `--skip-release-copy` – keep artefacts only under `dist/desktop/`.
- `--zip` – produce a ZIP archive next to the release directory for easy distribution.

## 3. Manual steps the script does not cover

- **Code signing**: apply your Authenticode/Notarization certificates to the generated binaries or installer.
- **Branding assets**: update icons and splash screens in `tradingbot_ibkr/assets/`.
- **Updater integration**: hook the build into your release pipeline or auto-updater if required.

## 4. Quick smoke test

After packaging, launch the binary once to verify dependencies were bundled correctly:

```bash
cd release/desktop/RomanBot
./RomanBot  # Windows: RomanBot.exe
```

Confirm the embedded FastAPI dashboard starts and the pywebview shell opens.

## 5. Troubleshooting

- Missing modules at runtime – ensure `pyinstaller` warnings in `build/desktop/warn-RomanBot.txt` are resolved.
- `ISCC.exe not found` – install Inno Setup and add it to `PATH`.
- macOS Gatekeeper blocks app – notarise the bundle or distribute via a signed DMG.
