"""Evidencias visuales e inferencia de demostracion; no modifica los modelos."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from entrenar_yuca import (
    ROOT, OUT, CLASSES, MEAN, STD, build_model, make_transform, read_manifest
)

PAIRS = [
    ("cbb", "healthy"), ("cbsd", "healthy"),
    ("cgm", "healthy"), ("cmd", "healthy"),
    ("cbsd", "cbb"), ("cgm", "cmd"),
    ("cmd", "cgm"), ("healthy", "cbsd"),
]


def load_model():
    model = build_model("E2", pretrained=False)
    model.load_state_dict(torch.load(OUT / "E2" / "mejor.pt",
                                     map_location="cpu", weights_only=True))
    # CPU para evitar dependencias de backward en MPS. No medir tiempos aqui.
    return model.eval()


def predict(model, image):
    x = make_transform("E2", False)(image).unsqueeze(0)
    with torch.no_grad():
        p = model(x).softmax(1)[0]
    return CLASSES[int(p.argmax())], float(p.max())


def gradcam(model, image):
    # Ultima capa convolucional (proyeccion final); mapa de la clase predicha.
    captured = {}
    def hook(module, inputs, output):
        captured["a"] = output
    handle = model.features[-1].register_forward_hook(hook)
    try:
        x = make_transform("E2", False)(image).unsqueeze(0).requires_grad_(True)
        scores = model(x)
        idx = int(scores.argmax(1).item())
        a = captured["a"]
        g = torch.autograd.grad(scores[0, idx], a)[0]
        cam = (a * g.mean(dim=(2, 3), keepdim=True)).sum(1).relu()[0]
        cam = cam.detach().cpu().numpy()
        maximum = float(cam.max())
        if maximum > 0:
            cam /= maximum
        return cam, CLASSES[idx], maximum > 0
    finally:
        handle.remove()


def evidence():
    read_manifest()
    model = load_model()
    errors = pd.read_csv(OUT / "E2" / "errores.csv")
    selected = []
    for a, b in PAIRS:
        subset = errors[(errors.real == a) & (errors.predicha == b)]
        if subset.empty:
            raise ValueError(f"Par sin evidencia: {a}, {b}")
        selected.append(subset.sort_values(["confianza", "archivo"],
                                            ascending=[False, True]).iloc[0])
    selected = pd.DataFrame(selected).reset_index(drop=True)
    # Directorio local: contiene imagenes del dataset, no publicar.
    dest = Path("evidencias_visuales_local")
    dest.mkdir(exist_ok=True)
    crop = transforms.Compose([transforms.Resize(256, antialias=True),
                               transforms.CenterCrop(224)])
    fig, axes = plt.subplots(8, 2, figsize=(8, 22))
    checked = []
    for i, row in selected.iterrows():
        path = ROOT / "test" / row.real / row.archivo
        with Image.open(path) as src:
            image = src.convert("RGB")
        pred, conf = predict(model, image)
        if pred != row.predicha:
            raise RuntimeError(f"Prediccion CPU difiere del registro MPS: {row.archivo}")
        checked.append({"archivo": row.archivo, "real": row.real,
                        "predicha_registro": row.predicha,
                        "confianza_registro_mps": row.confianza,
                        "predicha_cpu": pred, "confianza_cpu": conf,
                        "observacion_equipo": "", "causa_hipotesis_equipo": ""})
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{i+1}. {row.archivo}\nReal: {row.real} | Pred.: {row.predicha}",
                             fontsize=9)
        axes[i, 1].imshow(crop(image))
        axes[i, 1].set_title(f"Entrada visual 224x224 | score MPS {row.confianza:.3f}",
                             fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest / "8_errores_E2.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(checked).to_csv(dest / "8_errores_E2.csv", index=False)
    # Cuatro mapas de los casos de enfermedad -> sana.
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    for i, row in selected.iloc[:4].iterrows():
        with Image.open(ROOT / "test" / row.real / row.archivo) as src:
            image = src.convert("RGB")
        cam, pred, nonzero = gradcam(model, image)
        if pred != row.predicha:
            raise RuntimeError(f"Grad-CAM: prediccion no coincide en {row.archivo}")
        shown = crop(image)
        axes[i, 0].imshow(shown)
        axes[i, 1].imshow(shown)
        if nonzero:
            axes[i, 1].imshow(cam, cmap="jet", alpha=.45, interpolation="bilinear",
                               extent=(0, 224, 224, 0), vmin=0, vmax=1)
        axes[i, 0].set_title(f"Real: {row.real} | {row.archivo}", fontsize=9)
        axes[i, 1].set_title(f"Grad-CAM: {pred}" + ("" if nonzero else " (mapa nulo)"))
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest / "4_gradcam_E2.png", dpi=150)
    plt.close(fig)
    print("Evidencias locales:", dest.resolve())
    print("No publicar las imagenes sin confirmar licencia. No se recalcularon metricas de test.")


def demo(path):
    model = load_model()
    with Image.open(path) as src:
        image = src.convert("RGB")
    x = make_transform("E2", False)(image).unsqueeze(0)
    with torch.no_grad():
        p = model(x).softmax(1)[0]
    print(json.dumps({"archivo": str(path), "experimento": "E2",
                      "prediccion": CLASSES[int(p.argmax())],
                      "scores_softmax": {c: float(v) for c, v in zip(CLASSES, p)},
                      "nota": "No es diagnostico validado ni probabilidad calibrada."},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evidencias", action="store_true")
    group.add_argument("--imagen", type=Path)
    args = parser.parse_args()
    if args.evidencias:
        evidence()
    else:
        demo(args.imagen)
