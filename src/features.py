import hashlib

import librosa
import numpy as np

from . import config as C
from .data import fix_length, load_audio


def log_mel(y):
    y = fix_length(y)
    m = librosa.feature.melspectrogram(
        y=y,
        sr=C.SAMPLE_RATE,
        n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS,
        fmax=C.FMAX,
    )
    m = librosa.power_to_db(m, ref=np.max)
    return ((m - m.mean()) / (m.std() + 1e-6)).astype(np.float32)


def handcrafted(y):
    y = fix_length(y)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=C.SAMPLE_RATE,
        n_mfcc=20,
        n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH,
        fmax=C.FMAX,
    )
    delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH
    )[0]
    f0 = librosa.yin(
        y,
        fmin=150,
        fmax=1000,
        sr=C.SAMPLE_RATE,
        frame_length=1024,
        hop_length=C.HOP_LENGTH,
    )

    n = min(mfcc.shape[1], len(rms), len(f0), len(centroid), len(zcr))
    mfcc, delta, rms, centroid, zcr, f0 = (
        mfcc[:, :n],
        delta[:, :n],
        rms[:n],
        centroid[:n],
        zcr[:n],
        f0[:n],
    )
    active = rms > 0.1 * rms.max()
    if active.sum() < 2:
        active = np.ones(n, dtype=bool)

    feats = [
        mfcc[:, active].mean(axis=1),
        mfcc[:, active].std(axis=1),
        delta[:, active].mean(axis=1),
        delta[:, active].std(axis=1),
        [
            f0[active].mean(),
            f0[active].std(),
            np.median(f0[active]),
            f0[active].min(),
            f0[active].max(),
            rms[active].mean(),
            rms[active].std(),
            centroid[active].mean(),
            centroid[active].std(),
            zcr[active].mean(),
            active.mean(),
        ],
    ]
    return np.concatenate(feats).astype(np.float32)


def build_features(paths):
    key = hashlib.md5("|".join(p.name for p in paths).encode()).hexdigest()[:10]
    cache_file = C.CACHE_DIR / f"features_{key}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        return cached["mel"], cached["hand"]
    mels, hands = [], []
    for p in paths:
        y = load_audio(p)
        mels.append(log_mel(y))
        hands.append(handcrafted(y))
    mels, hands = np.stack(mels), np.stack(hands)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, mel=mels, hand=hands)
    return mels, hands
