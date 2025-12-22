from pathlib import Path

from munl.settings import DEFAULT_DEVICE
DATA_PATH = Path("/data/santiago.medina/deepunlearn")
ARTIFACTS_PATH = DATA_PATH / "artifacts"
FIXED_SPLITS_PATH = ARTIFACTS_PATH / "fixed_splits"
MODEL_INITIALIZATIONS_PATH = ARTIFACTS_PATH / "model_initializations"

DEFAULT_VAL_RATIO = 0.15
DEFAULT_FORGET_RATIO = 0.10
