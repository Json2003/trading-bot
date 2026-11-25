# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

HERE = Path(SPEC).resolve().parent

# Let PyInstaller's build hooks import the installed packages before checkout shims.
os.environ["DISABLE_STUBS"] = "1"
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != HERE.parent]

datas = [
    (str(HERE / directory), directory)
    for directory in ("templates", "static", "assets", "models", "datafiles")
    if (HERE / directory).is_dir()
]

a = Analysis(
    [str(HERE / "roman_bot.py")],
    pathex=[str(HERE), str(HERE.parent)],
    binaries=[],
    datas=datas,
    hiddenimports=["win32timezone", "win32com"],
    hookspath=[str(HERE / "pyinstaller_hooks")],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RomanBot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="RomanBot")
