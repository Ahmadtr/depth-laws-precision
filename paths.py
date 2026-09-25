"""Central configuration of data, model and result locations.

All scripts read their paths from this module. Override the defaults with environment variables:
    DL_DATA      directory for datasets and downloaded models   (default: ./data/cache)
    DL_RESULTS   directory for result files                      (default: ./results)
    DL_MODELS    directory for trained CNN/ViT checkpoints       (default: <DL_RESULTS>/models)
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DL_DATA", os.path.join(REPO, "data", "cache"))
RESULTS = os.environ.get("DL_RESULTS", os.path.join(REPO, "results"))
MODELS = os.environ.get("DL_MODELS", os.path.join(RESULTS, "models"))
HF = os.path.join(DATA, "hf")

for _d in (DATA, RESULTS, MODELS):
    os.makedirs(_d, exist_ok=True)


def result(name):
    """Path of a result file inside RESULTS (sub-directories are created on demand)."""
    p = os.path.join(RESULTS, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
