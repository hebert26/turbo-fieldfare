#!/usr/bin/env python3
"""Use the unchanged parent-stage memory and serial output checks."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import analyze

if __name__ == '__main__':
    raise SystemExit(analyze.main())
