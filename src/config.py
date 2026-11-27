from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_REPO_DIR = DATA_DIR / "donateacry-corpus"
CORPUS_DIR = CORPUS_REPO_DIR / "donateacry_corpus_cleaned_and_updated_data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CORPUS_URL = "https://github.com/gveres/donateacry-corpus.git"

SAMPLE_RATE = 8000
CLIP_SECONDS = 7.0
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
FMAX = 4000

CLASSES = ["hungry", "tired", "belly_pain", "burping"]
DISPLAY_NAMES = {
    "hungry": "Hungry",
    "tired": "Tired",
    "belly_pain": "In pain",
    "burping": "Needs to burp",
    "discomfort": "Discomfort",
}

EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-2
FOLDS = 5
SEED = 42
UNCERTAIN_BELOW = 0.5
