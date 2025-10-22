# Splitstar Operations Console Dashboard

This directory integrates multiple frontend clients and a backend server for
interacting with the Splitstar Operations Console runtime.

## Components

- **backend/** – Flask server that exposes API endpoints by reusing the trading bot's built-in dashboard.
- **electron-app/** – Desktop client implemented with Electron. It now ships with a
  standalone mode that simulates data plugins and exchange connectivity so the
  desktop dashboard can operate without the FastAPI backend. You can manage
  plugins, toggle exchange connectivity, stage market or limit orders, and
  inspect activity logs entirely on the local session.
- **flutter_app/** – Mobile client implemented with Flutter.

Each component is a minimal starter intended to be expanded for full dashboard functionality.
