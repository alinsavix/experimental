#!/usr/bin/env python3
"""Launcher script for ChArUco calibration application."""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Import and run main
from chalibrate.main import main

if __name__ == '__main__':
    main()
