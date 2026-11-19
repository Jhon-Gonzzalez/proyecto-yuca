"""Inventario del dataset: imágenes por clase y partición, y resoluciones."""
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd

RAIZ = Path("data/cassavaleafdata")
SPLITS = ["train", "validation", "test"]
EXT = {".jpg", ".jpeg", ".png"}

filas = []
for split in SPLITS:
    for carpeta_clase in sorted((RAIZ / split).iterdir()):
        if not carpeta_clase.is_dir():
            continue
        for img_path in carpeta_clase.iterdir():
            if img_path.suffix.lower() not in EXT:
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    modo = im.mode
                ok = True
            except Exception:
                w = h = None
                modo = None
                ok = False
            filas.append({
                "split": split,
                "clase": carpeta_clase.name,
                "archivo": img_path.name,
                "ancho": w,
                "alto": h,
                "modo": modo,
                "legible": ok,
            })

df = pd.DataFrame(filas)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/inventario.csv", index=False)

print("\nTotal de imágenes:", len(df))
print("\nImágenes por partición y clase:")
tabla = df.pivot_table(index="clase", columns="split", values="archivo",
                       aggfunc="count", fill_value=0)
tabla["total"] = tabla.sum(axis=1)
print(tabla)

print("\nProporción por clase en train (%):")
print((df[df.split == "train"].clase.value_counts(normalize=True) * 100).round(1))

print("\nImágenes ilegibles:", (~df.legible).sum())
print("\nResoluciones más comunes:")
print(Counter(zip(df.ancho, df.alto)).most_common(5))
print("\nModos de color:", df.modo.value_counts().to_dict())