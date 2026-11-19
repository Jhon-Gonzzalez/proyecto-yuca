"""Genera particiones limpias a partir de los resultados de duplicados.py.

Decisiones del equipo (ver bitácora):
 1. Fotos idénticas con etiqueta distinta -> se excluyen TODAS las copias
    (no se puede saber cuál etiqueta es la correcta).
 2. Duplicados entre particiones con la misma etiqueta -> se conserva la copia
    de train y se excluye la de validation/test (test honesto).
 3. Duplicados exactos dentro de la misma partición -> se conserva una copia.
"""
import pandas as pd

ORDEN = {"train": 0, "validation": 1, "test": 2}

inv = pd.read_csv("results/inventario.csv")
inv["id"] = inv.split + "/" + inv.clase + "/" + inv.archivo
pares = pd.read_csv("results/casi_duplicados.csv")
exactos = pd.read_csv("results/duplicados_exactos.csv")
exactos["id"] = exactos.split + "/" + exactos.clase + "/" + exactos.archivo


def ident(f, s):
    return f"{f[f'split_{s}']}/{f[f'clase_{s}']}/{f[f'archivo_{s}']}"


excluir = {}  # id de imagen -> motivo

# Regla 1: etiquetas contradictorias (se procesa primero para que tenga prioridad)
for _, f in pares[~pares.misma_clase].iterrows():
    for s in ("a", "b"):
        excluir.setdefault(ident(f, s), "etiqueta contradictoria")
for _, g in exactos.groupby("md5"):
    if g.clase.nunique() > 1:
        for i in g.id:
            excluir.setdefault(i, "etiqueta contradictoria")

# Regla 2: duplicados entre particiones con la misma etiqueta
for _, f in pares[pares.misma_clase].iterrows():
    a, b = ident(f, "a"), ident(f, "b")
    perdedor = a if ORDEN[f.split_a] > ORDEN[f.split_b] else b
    excluir.setdefault(perdedor, "duplicado de otra partición")

# Regla 3: duplicados exactos (misma clase) -> se queda la copia de la partición más temprana
for _, g in exactos.groupby("md5"):
    if g.clase.nunique() > 1:
        continue
    g = g.assign(o=g.split.map(ORDEN)).sort_values(["o", "archivo"])
    for i in g.id.iloc[1:]:
        excluir.setdefault(i, "duplicado exacto")

desconocidos = set(excluir) - set(inv.id)
if desconocidos:
    print("ADVERTENCIA: ids que no están en el inventario:", len(desconocidos))

inv["motivo"] = inv.id.map(excluir)
limpias = inv[inv.motivo.isna()]
excluidas = inv[inv.motivo.notna()]

limpias[["split", "clase", "archivo"]].to_csv("results/particiones_limpias.csv", index=False)
excluidas[["split", "clase", "archivo", "motivo"]].to_csv("results/excluidas.csv", index=False)

print(f"\nImágenes originales: {len(inv)}  |  excluidas: {len(excluidas)}  |  limpias: {len(limpias)}")
print("\nExcluidas por motivo:")
print(excluidas.motivo.value_counts())
print("\nExcluidas por partición:")
print(excluidas.split.value_counts())

antes = inv.pivot_table(index="clase", columns="split", values="archivo", aggfunc="count")
despues = limpias.pivot_table(index="clase", columns="split", values="archivo", aggfunc="count")
print("\nImágenes por clase y partición DESPUÉS de limpiar (entre paréntesis, cuántas se quitaron):")
print(despues.astype(str) + " (-" + (antes - despues).astype(int).astype(str) + ")")