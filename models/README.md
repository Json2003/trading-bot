Models directory
================

Purpose
- Store trained model artifacts and metadata produced by training scripts.

Layout
- checkpoints/: large binary checkpoints (.pt/.pth) for local use only.
- registry.json: small metadata registry written by training (created at runtime).

Notes
- Checkpoints are gitignored; `registry.json` can be committed if desired.

