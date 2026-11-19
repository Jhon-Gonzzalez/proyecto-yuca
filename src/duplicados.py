"""Detección de duplicados exactos y casi-duplicados entre particiones."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path("data/cassavaleafdata")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path, size=8):
    """Hash perceptual: compara cada píxel con su vecino en una versión 9x8 en grises."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()          # 64 bits
    return np.packbits(bits).view(">u8")[0]


def pares_cercanos(A, B, umbral):
    """Pares (a, b) cuyo dHash difiere en <= umbral bits."""
    hb = B["dhash"].to_numpy(dtype=np.uint64)
    filas = []
    for _, fa in A.iterrows():
        dist = np.bitwise_count(hb ^ np.uint64(fa["dhash"]))
        for j in np.where(dist <= umbral)[0]:
            fb = B.iloc[j]
            filas.append({
                "split_a": fa.split, "clase_a": fa.clase, "archivo_a": fa.archivo,
                "split_b": fb.split, "clase_b": fb.clase, "archivo_b": fb.archivo,
                "distancia": int(dist[j]),
                "misma_clase": fa.clase == fb.clase,
            })
    return filas


def ruta(fila, s):
    return RAIZ / fila[f"split_{s}"] / fila[f"clase_{s}"] / fila[f"archivo_{s}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=int, default=4,
                    help="Máx. bits distintos en dHash para considerar casi-duplicado")
    args = ap.parse_args()

    df = pd.read_csv("results/inventario.csv")
    md5s, dhs = [], []
    for i, f in enumerate(df.itertuples()):
        p = RAIZ / f.split / f.clase / f.archivo
        md5s.append(md5(p))
        dhs.append(dhash(p))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} imágenes procesadas")
    df["md5"] = md5s
    df["dhash"] = np.array(dhs, dtype=np.uint64)

    # 1) Duplicados exactos
    grupos = df.groupby("md5").filter(lambda g: len(g) > 1)
    entre = grupos.groupby("md5").filter(lambda g: g.split.nunique() > 1)
    print(f"\nDuplicados EXACTOS: {grupos.md5.nunique()} grupos "
          f"({len(grupos)} archivos); entre particiones distintas: {entre.md5.nunique()} grupos")
    grupos.sort_values("md5").to_csv("results/duplicados_exactos.csv", index=False)

    # 2) Casi-duplicados entre particiones
    filas = []
    for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]:
        par = pares_cercanos(df[df.split == a], df[df.split == b], args.umbral)
        print(f"Casi-duplicados {a} vs {b} (umbral={args.umbral}): {len(par)} pares")
        filas += par
    pares = pd.DataFrame(filas)
    pares.to_csv("results/casi_duplicados.csv", index=False)

    if len(pares):
        print("Pares con etiqueta DISTINTA:", (~pares.misma_clase).sum())
        # Figura para inspección visual de los 12 pares más parecidos
        muestra = pares.sort_values("distancia").head(12)
        fig, ax = plt.subplots(len(muestra), 2, figsize=(7, 3 * len(muestra)))
        ax = np.atleast_2d(ax)
        for k, (_, f) in enumerate(muestra.iterrows()):
            for c, s in enumerate(["a", "b"]):
                ax[k, c].imshow(Image.open(ruta(f, s)))
                ax[k, c].set_title(f"{f[f'split_{s}']}/{f[f'clase_{s}']}\n"
                                   f"{f[f'archivo_{s}']}  d={f.distancia}", fontsize=7)
                ax[k, c].axis("off")
        plt.tight_layout()
        plt.savefig("results/casi_duplicados_muestra.png", dpi=110)
        print("Figura guardada en results/casi_duplicados_muestra.png")


if __name__ == "__main__":
    main()