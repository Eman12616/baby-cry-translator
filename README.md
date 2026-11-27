# Baby Cry Translator

Predicts why a baby is crying (hungry, tired, in pain, needs to burp) from a short audio clip.
Python, librosa log-mel spectrograms, and a small CNN, compared against an MFCC + SVM baseline.

## Quick start

```bash
pip install -r requirements.txt
python -m src.train all
python app.py
```

`python -m src.train all` downloads the Donate-A-Cry corpus, runs 5-fold cross-validation for both models, then trains the final CNN on all clips and saves it to `models/cnn.pt`.
Results are written to `results/metrics.json` and `results/confusion_matrices.png`.

Other commands:

```bash
python -m src.train cv
python -m src.train final
python -m src.train cv --classes hungry tired belly_pain burping discomfort
python -m src.predict path/to/cry.wav
```

## How it works

1. Audio is resampled to 8 kHz mono and padded or cropped to 7 seconds.
2. Clips become 64-band log-mel spectrograms (librosa).
3. The CNN has 4 conv blocks with global average pooling. Training uses class-weighted loss, label smoothing, and augmentation (time shift, SpecAugment masks, noise).
4. The baseline is an RBF SVM on MFCC, delta-MFCC, F0, energy, and spectral statistics.
5. Longer recordings are split into 7-second windows and averaged. If top confidence is below 50%, the app says "not sure" instead of forcing a label.

## Evaluation

Splits are grouped by baby: the app/user ID in each filename is used so the same baby never appears in both train and test (5-fold StratifiedGroupKFold). Metric is macro-F1, since accuracy on this data is dominated by the "hungry" class.

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Majority class (always "hungry") | 0.235 | 0.888 |
| MFCC + SVM | 0.234 | 0.879 |
| Log-mel CNN | 0.191 | 0.528 |

430 clips, 217 babies. Class counts: hungry 382, tired 24, belly pain 16, burping 8.

## Honest limitations

- Neither model beats always guessing "hungry" on macro-F1 when babies are held out. With 8 burping clips and 16 belly-pain clips, there is not enough data to learn those classes, and labels come from parents using a phone app.
- Treat the demo as a research prototype. It is not a medical or parenting tool.
- Random (non-grouped) splits would look much better because the same baby leaks into train and test. That number would be misleading.

## Ideas to improve it

- Replace the from-scratch CNN with embeddings from a pretrained audio model (YAMNet, AST, wav2vec2) plus a linear classifier.
- Add more labeled data (Dunstan Baby Language, ChatterBaby, or your own recordings).
- Merge to fewer classes (for example hungry vs. not hungry, or pain vs. not pain) and report calibrated confidence.

## Data

Donate-A-Cry corpus: https://github.com/gveres/donateacry-corpus
