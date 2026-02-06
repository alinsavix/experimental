#!/bin/bash
# Test GUI calibration with correct board config
uv run chalibrate.py \
  --images-dir ./board_images \
  --squares-x 8 \
  --squares-y 8 \
  --square-length 30.0 \
  --marker-length 20.0 \
  --dict DICT_4X4_50
