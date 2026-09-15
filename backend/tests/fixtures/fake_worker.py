#!/usr/bin/env python3
"""Fake worker for testing translation.py's subprocess IPC mechanism for
real, without needing torch/transformers/IndicTransToolkit. Mimics the
exact protocol translation_worker.py uses. Mode is read from an env var
(FAKE_WORKER_MODE) so the real _ensure_worker() call signature never needs
test-only args."""
import json
import os
import sys

mode = os.environ.get("FAKE_WORKER_MODE", "normal")

if mode == "crash_on_startup":
    print(json.dumps({"ready": False, "error": "simulated startup crash"}), flush=True)
    sys.exit(1)

if mode == "hang_on_startup":
    import time
    time.sleep(9999)

print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    sentences = req.get("sentences", [])
    if mode == "crash_on_request":
        sys.exit(1)
    if mode == "bad_json":
        print("not json", flush=True)
        continue
    translated = [f"[MR:{s}]" for s in sentences]
    print(json.dumps({"ok": True, "translations": translated}), flush=True)
