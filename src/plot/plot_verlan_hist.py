#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated wrapper for plot_probability_histogram.

This entrypoint is kept for backward compatibility. Please use:
  python src/plot/plot_probability_histogram.py [args]
"""

from .plot_probability_histogram import main

if __name__ == "__main__":
    raise SystemExit(main())

