"""``python -m pyising``."""

import os
import sys

from .cli import main

code = main()
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)  # skip GPU-runtime teardown (see bench/run.py)
