#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper to preserve the original entrypoint name.

Usage stays the same:
  python align_and_mean_movement_v1.py [args...]
"""

from align_mean_movement.cli import main

if __name__ == "__main__":
    main()
