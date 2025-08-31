"""Desktop launcher for the TradingBot dashboard using pywebview.

Starts the Flask dashboard in a background process and opens a native window.
"""
import threading
import time
import webview
from pathlib import Path
import sys
import os
import pystray
from PIL import Image, ImageDraw
import shutil
try:
    from win32com.client import Dispatch
except Exception:
    Dispatch = None

# Single-process launcher: run the Flask dashboard in a background thread
# and embed it in a native window using pywebview.
ROOT = Path(__file__).resolve().parents[0]
ICON = ROOT / 'assets' / 'centurion.svg'

STARTUP_SHORTCUT = Path(os.getenv('APPDATA')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' / 'Startup' / 'Roman Bot.lnk'

def start_flask_app():
    # import here so module imports happen after working dir is set
    import dashboard
    # run Flask app; set host to 127.0.0.1 for local-only
    dashboard.app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

def main():
    # start Flask in a daemon thread
    t = threading.Thread(target=start_flask_app, daemon=True)
    t.start()
    # wait a short while for server to start
    time.sleep(1.5)
    window = webview.create_window('Roman Bot', 'http://127.0.0.1:5001', icon=str(ICON))

    # create tray icon
    def create_image():
        # create a simple PIL image for tray (fallback if SVG not supported)
        img = Image.new('RGBA', (64, 64), (139,0,0,255))
        d = ImageDraw.Draw(img)
        d.ellipse((8,8,56,56), fill=(255,215,0,255))
        return img

    def on_show(icon, item):
        try:
            webview.windows[0].show()
        except Exception:
            pass

    def on_hide(icon, item):
        try:
            webview.windows[0].hide()
        except Exception:
            pass

    def create_startup_shortcut():
        if Dispatch is None:
            return False
        try:
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(STARTUP_SHORTCUT))
            # prefer pythonw to avoid console window, fall back to python
            python_exec = Path(sys.executable)
            pythonw = python_exec.with_name('pythonw.exe')
            exe = str(pythonw) if pythonw.exists() else str(python_exec)
            shortcut.Targetpath = exe
            # pass the launcher script as an argument
            shortcut.Arguments = f'"{str(Path(__file__).resolve())}"'
            shortcut.WorkingDirectory = str(ROOT)
            # icon expects path,index; use the centurion svg if possible (may not be supported)
            shortcut.IconLocation = str(ICON)
            shortcut.WindowStyle = 1
            shortcut.save()
            return True
        except Exception:
            return False

    def remove_startup_shortcut():
        try:
            if STARTUP_SHORTCUT.exists():
                STARTUP_SHORTCUT.unlink()
                return True
        except Exception:
            pass
        return False

    def on_toggle_autostart(icon, item):
        # toggle: create if missing, remove if exists
        if STARTUP_SHORTCUT.exists():
            ok = remove_startup_shortcut()
            # optionally notify
        else:
            ok = create_startup_shortcut()

    def on_exit(icon, item):
        icon.stop()
        try:
            for w in webview.windows:
                w.destroy()
        except Exception:
            pass
        os._exit(0)

    menu = pystray.Menu(pystray.MenuItem('Show', on_show), pystray.MenuItem('Hide', on_hide), pystray.MenuItem('Toggle Autostart', on_toggle_autostart), pystray.MenuItem('Exit', on_exit))
    icon = pystray.Icon('roman_bot', create_image(), 'Roman Bot', menu)
    # start tray in separate thread so webview loop runs
    tray_thread = threading.Thread(target=icon.run, daemon=True)
    tray_thread.start()

    webview.start()

if __name__ == '__main__':
    main()
