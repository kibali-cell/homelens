"""Central configuration: paths and constants shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "housing.csv"
FIG_DIR = ROOT / "reports" / "figures"
OUT_DIR = ROOT / "reports" / "outputs"
ART_DIR = ROOT / "artifacts"
RESULTS_PATH = OUT_DIR / "results.json"

RANDOM_STATE = 42
TARGET = "median_house_value"
TEST_SIZE = 0.2
PREMIUM_QUANTILE = 0.75   # a district is "premium" if its value is in the top quartile of the TRAIN set
N_TIERS = 4               # price tiers = quartiles of the TRAIN target

for _d in (FIG_DIR, OUT_DIR, ART_DIR):
    _d.mkdir(parents=True, exist_ok=True)
