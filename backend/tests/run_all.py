"""Convenience runner: executes every test file's standalone runner and
reports a combined summary. Real deployments should just use `pytest`;
this exists for environments where installing pytest isn't convenient.

Usage:  python tests/run_all.py
"""
import subprocess
import sys
from pathlib import Path

TEST_FILES = [
    "test_references.py",
    "test_ingest_detection.py",
    "test_store_and_auth.py",
    "test_retrieval_and_export.py",
    "test_translation.py",
    "test_ocr_preprocessing.py",
    "test_scheduler.py",
    "test_inline_schedule.py",
]

if __name__ == "__main__":
    here = Path(__file__).parent
    total_failed = 0
    for f in TEST_FILES:
        print(f"\n{'=' * 60}\n{f}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(here / f)])
        if result.returncode != 0:
            total_failed += 1
    print(f"\n{'=' * 60}")
    if total_failed:
        print(f"❌ {total_failed}/{len(TEST_FILES)} test file(s) had failures")
        sys.exit(1)
    else:
        print(f"✅ All {len(TEST_FILES)} test files passed")
