import requests
import torch
import numpy as np
from pathlib import Path

# 1. Verify environment compatibility
print(f"Using PyTorch version: {torch.__version__}")
print(f"Using NumPy version: {np.__version__}")

# 2. Download helper functions if they don't exist
file_name = "helper.py"  # Standardized name
if Path(file_name).is_file():
    print(f"{file_name} already exists, skipping download.")
else:
    print(f"Downloading {file_name}...")
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    request = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(request.content)

# 3. Import from the helper file
# Note: Ensure the string matches the filename above (minus .py)
try:
    from helper import plot_predictions, plot_decision_boundary

    print("Helper functions imported successfully!")
except ImportError as e:
    print(f"Import failed: {e}")
