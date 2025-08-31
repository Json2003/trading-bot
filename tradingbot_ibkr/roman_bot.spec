# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

block_cipher = None

datas = []
for d in ['templates', 'static', 'assets', 'models', 'datafiles']:
    src = str(HERE / d)
    if os.path.exists(src):
        for root, dirs, files in os.walk(src):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, src)
                datas.append((full, os.path.join(d, rel)))

hiddenimports = ['win32timezone', 'win32com', 'pkg_resources.py2_warn']

a = Analysis(['roman_bot.py'],
             pathex=[str(HERE)],
             binaries=[],
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='RomanBot', debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=False )
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, upx_exclude=[], name='RomanBot')
