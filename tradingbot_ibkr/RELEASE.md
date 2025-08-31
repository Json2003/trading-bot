RomanBot Release build
======================

This folder contains helper scripts to produce a ZIP release and an optional Windows installer for the desktop app.

Prereqs
- Python and project dependencies installed in an environment (recommended: virtualenv).
- PyInstaller installed (`pip install pyinstaller`).
- (Optional) Inno Setup installed to build a native .exe installer (ISCC.exe on PATH).

Quick build (PowerShell, run from `tradingbot_ibkr` directory):

```powershell
# Build single-file EXE (calls build_exe.ps1)
.\build_release.ps1 -Entry roman_bot.py -Name RomanBot -Version 0.1.0
```

Output
- `releases/` will contain a timestamped folder plus a ZIP with the EXE and optional model/data.
- If Inno Setup (ISCC) is installed, an installer `.exe` named like `RomanBotInstaller.exe` will be placed into `releases`.

Notes
- The built EXE is single-file and bundles templates/static assets via PyInstaller add-data lines in `build_exe.ps1`.
- Model artifacts (`model_store`) and `datafiles` are optionally copied into the release folder; include them only if desired.
- Test the built EXE on a VM or clean machine before distributing.
