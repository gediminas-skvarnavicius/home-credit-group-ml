import joblib
from pathlib import Path

__version__ = "0.1"
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(
    f"{BASE_DIR}/fee_model-{__version__}.joblib",
    "rb",
) as f:
    model_fees = joblib.load(f)
