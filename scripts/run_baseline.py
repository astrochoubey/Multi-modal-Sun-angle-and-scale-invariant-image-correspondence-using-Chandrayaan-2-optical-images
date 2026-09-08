"""
Script to execute baseline lunar image registration on the sample dataset.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from main import main

if __name__ == "__main__":
    main()
