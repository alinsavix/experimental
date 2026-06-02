@echo off
uv run python -m uvicorn server.main:app --host 0.0.0.0 --port 8084
