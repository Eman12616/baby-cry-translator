import gradio as gr
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.features import log_mel
from src.predict import load_model, predict, verdict

model, classes = load_model()


def spectrogram_figure(window):
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(
        log_mel(window),
        sr=C.SAMPLE_RATE,
        hop_length=C.HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmax=C.FMAX,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+.1f")
    ax.set_title("Log-mel spectrogram")
    fig.tight_layout()
    return fig


def analyze(audio_path):
    if audio_path is None:
        return None, "Upload or record a cry first.", None
    probs, window = predict(audio_path, model, classes)
    labelled = {C.DISPLAY_NAMES.get(c, c): p for c, p in probs.items()}
    fig = spectrogram_figure(window)
    plt.close(fig)
    return labelled, verdict(probs), fig


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Baby cry"),
    outputs=[
        gr.Label(num_top_classes=len(classes), label="Probabilities"),
        gr.Markdown(label="Result"),
        gr.Plot(label="What the model sees"),
    ],
    title="Baby Cry Translator",
    description=(
        "Predicts why a baby might be crying: hungry, tired, in pain, or needs to burp. "
        "Research prototype trained on a small public dataset. Not a medical tool."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
