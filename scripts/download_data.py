"""
Data preparation script for Lunar Image Registration.
Generates synthetic lunar craters simulating Chandrayaan-2 optical sensors.
"""

from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.generate_sample_data import generate_dataset

if __name__ == "__main__":
    print("Preparing lunar imagery dataset...")
    src, ref = generate_dataset()
    print("Dataset preparation complete.")
