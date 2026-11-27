import numpy as np
import torch

from . import config as C
from .data import load_audio
from .features import log_mel
from .models import CryCNN


def load_model(path=None):
    ckpt = torch.load(path or C.MODEL_DIR / "cnn.pt", map_location="cpu")
    model = CryCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def split_windows(y):
    n = int(C.SAMPLE_RATE * C.CLIP_SECONDS)
    if len(y) <= n:
        return [y]
    windows = [y[i : i + n] for i in range(0, len(y), n) if len(y[i : i + n]) >= n // 2]
    return windows or [y[:n]]


@torch.no_grad()
def predict(audio_path, model, classes):
    y = load_audio(audio_path)
    if len(y) == 0:
        raise ValueError("Audio file is empty")
    windows = split_windows(y)
    x = torch.from_numpy(np.stack([log_mel(w) for w in windows])).unsqueeze(1)
    probs = torch.softmax(model(x), dim=1).mean(dim=0).numpy()
    return {c: float(p) for c, p in zip(classes, probs)}, windows[0]


def verdict(probs):
    best = max(probs, key=probs.get)
    name = C.DISPLAY_NAMES.get(best, best)
    if probs[best] < C.UNCERTAIN_BELOW:
        return f"Not sure. Closest guess is **{name}**, but confidence is only {probs[best]:.0%}."
    return f"Most likely: **{name}** ({probs[best]:.0%})"


if __name__ == "__main__":
    import sys

    model, classes = load_model()
    probs, _ = predict(sys.argv[1], model, classes)
    print(verdict(probs))
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {C.DISPLAY_NAMES.get(c, c):<16}{p:.1%}")
