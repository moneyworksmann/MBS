#!/usr/bin/env python3
"""Entrypoint script for MBS financial calculator.

Usage:
    python3 run_mbs.py

This script performs two actions automatically:
  1. Runs the Python backend demo from `Python Script.py`.
  2. Starts a local HTTP server and opens `HTML script.html` in a browser.

No user prompts are required.
"""

import os
import sys
import subprocess
import time
import webbrowser

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PY_SCRIPT = os.path.join(REPO_ROOT, "Python Script.py")
HTML_FILE = os.path.join(REPO_ROOT, "HTML script.html")


def run_python_demo():
    print("\n=== Running Python backend demo (Python Script.py) ===")
    if not os.path.exists(PY_SCRIPT):
        raise FileNotFoundError(f"Python backend script not found: {PY_SCRIPT}")
    result = subprocess.run([sys.executable, PY_SCRIPT], cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Python script exited with status {result.returncode}")
    else:
        print("[OK] Python demo completed successfully")


def serve_html():
    if not os.path.exists(HTML_FILE):
        raise FileNotFoundError(f"HTML file not found: {HTML_FILE}")

    # Open the HTML file directly (no local webserver) to avoid directory prompt.
    file_url = 'file://' + os.path.abspath(HTML_FILE).replace(' ', '%20')
    print(f"\n=== Opening HTML UI at {file_url} ===")

    try:
        webbrowser.open(file_url)
        print("[INFO] Browser open request sent.")
    except Exception as e:
        print(f"[WARN] Could not open browser automatically: {e}")

    print("Open the URL above manually if needed.")
    return None


def main():
    try:
        run_python_demo()
        serve_html()

        # Give user time to view output; then exit cleanly.
        print("\nRunner complete. Press Ctrl+C to exit if still running.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting.")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == '__main__':
    main()
