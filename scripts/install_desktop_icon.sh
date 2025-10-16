#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Splitstar Trading Bot"
APP_ID="splitstar-trading-bot"

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# Find AppImage built by electron-builder
APPIMAGE_SRC="$(ls -1 "${REPO_DIR}/dashboard/electron-app/dist"/*AppImage 2>/dev/null | head -n1 || true)"
if [[ -z "${APPIMAGE_SRC}" ]]; then
  echo "No AppImage found in dashboard/electron-app/dist. Build it first:"
  echo "  cd ${REPO_DIR}/dashboard/electron-app && npm install && npm run build"
  exit 1
fi

# Install destinations
BIN_DIR="${HOME}/.local/bin"
APPIMAGE_DST="${BIN_DIR}/${APP_ID}"
LAUNCHER_DST="${BIN_DIR}/${APP_ID}-launch"
DESKTOP_DIR="${HOME}/.local/share/applications"
# Resolve the user's Desktop directory (if it exists)
USER_DIRS_FILE="${HOME}/.config/user-dirs.dirs"
USER_DESKTOP_DIR="${HOME}/Desktop"
if [[ -f "${USER_DIRS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${USER_DIRS_FILE}" || true
  if [[ -n "${XDG_DESKTOP_DIR:-}" ]]; then
    # Expand $HOME in the value
    USER_DESKTOP_DIR="${XDG_DESKTOP_DIR/#\$HOME/${HOME}}"
  fi
fi
ICON_DIR_SVG="${HOME}/.local/share/icons/hicolor/scalable/apps"
ICON_SRC_SVG="${REPO_DIR}/assets/splitstar_logo.svg"
ICON_DST_SVG="${ICON_DIR_SVG}/${APP_ID}.svg"
DESKTOP_FILE="${DESKTOP_DIR}/${APP_ID}.desktop"

mkdir -p "${BIN_DIR}" "${DESKTOP_DIR}" "${ICON_DIR_SVG}"

# Copy AppImage and make executable
install -m 0755 -D "${APPIMAGE_SRC}" "${APPIMAGE_DST}"

# Install icon (SVG scalable)
install -m 0644 -D "${ICON_SRC_SVG}" "${ICON_DST_SVG}"

# Create launcher wrapper that starts backend on 9000 if not running
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo ''
  echo 'APP_ID="splitstar-trading-bot"'
  echo "REPO_DIR=\"${REPO_DIR}\""
  cat <<'EOS'

# Detect if port 9000 is in use
is_listening() {
  (command -v ss &>/dev/null && ss -ltn | grep -q ":9000 ") || \
  (command -v netstat &>/dev/null && netstat -ltn 2>/dev/null | grep -q ":9000 ") || \
  (command -v nc &>/dev/null && nc -z 127.0.0.1 9000 2>/dev/null)
}

if ! is_listening; then
  # Prefer venv if present
  PYBIN="python3"
  if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    PYBIN="${REPO_DIR}/.venv/bin/python"
  fi
  (
    cd "${REPO_DIR}" && \
    nohup "${PYBIN}" -m uvicorn server:app --host 0.0.0.0 --port 9000 >> server.log 2>&1 &
    echo $! > server.pid
  ) || true
fi

exec "${HOME}/.local/bin/${APP_ID}" "$@"
EOS
} > "${LAUNCHER_DST}"
chmod +x "${LAUNCHER_DST}"

# Write desktop entry
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=${APP_NAME}
Comment=Launch the Splitstar Trading Bot dashboard and backend
Exec=${LAUNCHER_DST}
Icon=${APP_ID}
Terminal=false
Categories=Finance;Network;Utility;
StartupWMClass=Trading Bot Dashboard
EOF

# Attempt to refresh desktop icon cache (best-effort)
if command -v update-desktop-database &>/dev/null; then
  update-desktop-database "${DESKTOP_DIR}" || true
fi
if command -v gtk-update-icon-cache &>/dev/null; then
  gtk-update-icon-cache -f "${HOME}/.local/share/icons/hicolor" || true
fi

# Optionally place a desktop shortcut on the actual Desktop folder
if [[ -d "${USER_DESKTOP_DIR}" ]]; then
  DESKTOP_SHORTCUT="${USER_DESKTOP_DIR}/${APP_ID}.desktop"
  cp -f "${DESKTOP_FILE}" "${DESKTOP_SHORTCUT}"
  chmod +x "${DESKTOP_SHORTCUT}"
  # Mark as trusted for GNOME (if available)
  if command -v gio &>/dev/null; then
    gio set -t string "${DESKTOP_SHORTCUT}" metadata::trusted true || true
  fi
  echo "Desktop shortcut installed: ${DESKTOP_SHORTCUT}"
else
  echo "No Desktop folder detected at ${USER_DESKTOP_DIR}; skipping desktop shortcut."
fi

echo "Installed desktop launcher: ${DESKTOP_FILE}"
echo "Icon installed: ${ICON_DST_SVG}"
echo "Launcher: ${LAUNCHER_DST}"
echo "AppImage: ${APPIMAGE_DST}"
