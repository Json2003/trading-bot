# Workspace Diagnostics

The following commands were executed to inspect the repository state:

```bash
pwd
ls -la
git status
git rev-parse --abbrev-ref HEAD
```

These commands confirm the repository root at `/workspace/trading-bot`, list the contents of the directory, display a clean working tree, and identify the current branch as `work`.

## CI failure analysis

To reproduce the failing GitHub Actions job locally, the static analysis step from `.github/workflows/ci.yml` was executed with `ruff`.

```bash
ruff check .
```

This command currently reports hundreds of lint violations spread across the legacy `tradingbot_ibkr` utilities and other helper modules. Because the CI workflow executes `ruff check .` on every pull request, these pre-existing violations cause the workflow to fail before tests can run, leading to the systematic failure of all pull requests.
