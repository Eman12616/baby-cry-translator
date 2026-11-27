import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .data import list_clips
from .features import build_features
from .models import CryCNN


def augment(x):
    x = np.roll(x, np.random.randint(-40, 41), axis=1)
    for _ in range(2):
        f = np.random.randint(0, 8)
        f0 = np.random.randint(0, x.shape[0] - f)
        x[f0 : f0 + f, :] = 0
        t = np.random.randint(0, 30)
        t0 = np.random.randint(0, x.shape[1] - t)
        x[:, t0 : t0 + t] = 0
    return x + np.random.normal(0, 0.05, x.shape).astype(np.float32)


class MelDataset(Dataset):
    def __init__(self, mels, labels, train):
        self.mels, self.labels, self.train = mels, labels, train

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, i):
        x = self.mels[i].copy()
        if self.train:
            x = augment(x)
        return torch.from_numpy(x).unsqueeze(0), int(self.labels[i])


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(mels, labels, n_classes, epochs, device):
    model = CryCNN(n_classes).to(device)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights ** 0.75, device=device), label_smoothing=0.1
    )
    loader = DataLoader(
        MelDataset(mels, labels, train=True),
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        drop_last=len(mels) > C.BATCH_SIZE,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=C.LEARNING_RATE, epochs=epochs, steps_per_epoch=len(loader)
    )
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict_proba(model, mels, device):
    model.eval()
    out = []
    for i in range(0, len(mels), 64):
        x = torch.from_numpy(mels[i : i + 64]).unsqueeze(1).to(device)
        out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def summarize(name, y_true, y_pred, classes):
    labels = list(range(len(classes)))
    return {
        "model": name,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes, zero_division=0, output_dict=True
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusions(results, classes, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        cm = np.array(res["confusion"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{res['model']} (macro-F1 {res['macro_f1']:.2f})")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cross_validate(classes, epochs):
    paths, labels, groups = list_clips(classes)
    mels, hand = build_features(paths)
    device = pick_device()
    folds = StratifiedGroupKFold(n_splits=C.FOLDS, shuffle=True, random_state=C.SEED)

    pred_svm = np.zeros(len(labels), dtype=int)
    pred_cnn = np.zeros(len(labels), dtype=int)
    for k, (tr, te) in enumerate(folds.split(mels, labels, groups), 1):
        svm = make_pipeline(
            StandardScaler(), SVC(C=10, kernel="rbf", class_weight="balanced")
        )
        svm.fit(hand[tr], labels[tr])
        pred_svm[te] = svm.predict(hand[te])

        model = train_cnn(mels[tr], labels[tr], len(classes), epochs, device)
        pred_cnn[te] = predict_proba(model, mels[te], device).argmax(axis=1)
        print(f"fold {k}/{C.FOLDS} done", flush=True)

    majority = np.full(len(labels), np.bincount(labels).argmax())
    results = [
        summarize("Majority class", labels, majority, classes),
        summarize("MFCC + SVM", labels, pred_svm, classes),
        summarize("Log-mel CNN", labels, pred_cnn, classes),
    ]
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_confusions(results[1:], classes, C.RESULTS_DIR / "confusion_matrices.png")

    print(f"\nclips: {len(labels)}  babies: {len(set(groups))}  classes: {classes}")
    print(f"{'model':<16}{'macro-F1':>10}{'accuracy':>10}")
    for r in results:
        print(f"{r['model']:<16}{r['macro_f1']:>10.3f}{r['accuracy']:>10.3f}")
    for r in results[1:]:
        print(f"\n{r['model']} per-class F1:")
        for c in classes:
            rep = r["report"][c]
            print(f"  {c:<12} f1={rep['f1-score']:.2f}  recall={rep['recall']:.2f}  n={int(rep['support'])}")


def train_final(classes, epochs):
    paths, labels, _ = list_clips(classes)
    mels, _ = build_features(paths)
    device = pick_device()
    model = train_cnn(mels, labels, len(classes), epochs, device)
    C.MODEL_DIR.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.cpu().state_dict(), "classes": classes},
        C.MODEL_DIR / "cnn.pt",
    )
    print(f"saved {C.MODEL_DIR / 'cnn.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cv", "final", "all"])
    parser.add_argument("--classes", nargs="+", default=C.CLASSES)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = parser.parse_args()

    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    if args.mode in ("cv", "all"):
        cross_validate(args.classes, args.epochs)
    if args.mode in ("final", "all"):
        train_final(args.classes, args.epochs)


if __name__ == "__main__":
    main()
