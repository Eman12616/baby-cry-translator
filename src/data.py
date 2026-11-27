import subprocess

import librosa
import numpy as np

from . import config as C


def ensure_corpus():
    if C.CORPUS_DIR.exists():
        return
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", C.CORPUS_URL, str(C.CORPUS_REPO_DIR)],
        check=True,
    )


def list_clips(classes):
    ensure_corpus()
    paths, labels, groups = [], [], []
    for idx, name in enumerate(classes):
        for wav in sorted((C.CORPUS_DIR / name).glob("*.wav")):
            paths.append(wav)
            labels.append(idx)
            groups.append("-".join(wav.stem.split("-")[:5]))
    return paths, np.array(labels), np.array(groups)


def load_audio(path):
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE, mono=True)
    return y


def fix_length(y, seconds=C.CLIP_SECONDS):
    n = int(C.SAMPLE_RATE * seconds)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))
