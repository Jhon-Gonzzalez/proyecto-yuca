"""Extract reproducible descriptive EDA evidence for the cassava project.
Run from the project root: python -u eda_yuca_completo.py
Image appearance statistics and plots use training images only.
"""
from pathlib import Path
import hashlib
import json
import platform
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

ROOT = Path('data/cassavaleafdata')
RESULTS = Path('results')
OUT = RESULTS / 'eda_completo'
MANIFEST = RESULTS / 'particiones_limpias.csv'
CLASSES = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
SPLITS = ['train', 'validation', 'test']
SEED = 42  # sampling for figures only
COLS = ['ancho', 'alto', 'megapixeles', 'aspecto', 'brillo', 'contraste']

def save_table(df, name, index=False):
    df.to_csv(OUT / name, index=index)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, dtype=str)
    required = ['split', 'clase', 'archivo']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Columnas inesperadas: {df.columns.tolist()}')
    df = df[required].copy()
    missing = df.isna() | df.fillna('').apply(lambda x: x.str.strip().eq(''))
    quality = {'filas': len(df), 'campos_faltantes': missing.sum().to_dict(),
               'filas_duplicadas_manifest': int(df.duplicated().sum()),
               'split_invalidos': int((~df.split.isin(SPLITS)).sum()),
               'clases_invalidas': int((~df.clase.isin(CLASSES)).sum()),
               'sha256_manifest': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               'python': platform.python_version(), 'pandas': pd.__version__,
               'numpy': np.__version__, 'semilla_muestras': SEED,
               'brillo_contraste': 'Media y desviacion poblacional de intensidad en escala L (0-255), imagen completa, sin redimensionar. Solo train.'}
    print('Filas del manifiesto:', len(df))
    table = pd.crosstab(df.clase, df.split).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
    table['total'] = table.sum(axis=1)
    save_table(table, '01_conteos.csv', True)
    percent = table.div(table.sum(axis=0)).mul(100)
    save_table(percent, '02_porcentajes.csv', True)
    rows, errors = [], []
    for i, r in enumerate(df.itertuples(index=False)):
        if missing.iloc[i].any() or r.split not in SPLITS or r.clase not in CLASSES:
            errors.append(dict(split=r.split, clase=r.clase, archivo=r.archivo, error='Registro invalido'))
            continue
        path = ROOT / r.split / r.clase / r.archivo
        try:
            with Image.open(path) as img:
                w, h = img.size
                mode, fmt = img.mode, img.format
                img.load()  # checks that the actual pixels are decodable
                rec = dict(split=r.split, clase=r.clase, archivo=r.archivo,
                           ancho=w, alto=h, modo=mode, formato=fmt,
                           megapixeles=w*h/1e6, aspecto=w/h, bytes=path.stat().st_size)
                if r.split == 'train':
                    gray = np.asarray(img.convert('L'), dtype=np.float32)
                    rec.update(brillo=float(gray.mean(dtype=np.float64)),
                               contraste=float(gray.std(dtype=np.float64)))
                rows.append(rec)
        except Exception as exc:
            errors.append(dict(split=r.split, clase=r.clase, archivo=r.archivo, error=str(exc)))
        if (i+1) % 500 == 0:
            print(f'Procesadas {i+1}/{len(df)}', flush=True)
    images = pd.DataFrame(rows)
    save_table(pd.DataFrame(errors, columns=['split','clase','archivo','error']), '03_errores.csv')
    quality['archivos_leidos'] = len(images)
    quality['registros_con_error'] = len(errors)
    (OUT / '04_calidad.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding='utf-8')
    if images.empty:
        raise RuntimeError(f'No se pudo leer ninguna imagen. Revisar {OUT / "03_errores.csv"}')
    save_table(images, '05_atributos_imagenes.csv')
    train = images.loc[images.split.eq('train')].copy()
    if train.empty:
        raise RuntimeError('No se encontraron imagenes de train')
    cat = []
    for c in ['split','clase','modo','formato']:
        for v,n in images[c].value_counts(dropna=False).items():
            cat.append(dict(atributo=c, categoria=v, frecuencia=n, porcentaje=n/len(images)*100))
    save_table(pd.DataFrame(cat), '06_categoricos.csv')
    stats = train[COLS].agg(['count','mean','median','min','max','std']).T
    save_table(stats, '07_numericos_train.csv', True)
    grouped = train.groupby('clase')[COLS].agg(['count','mean','median','std','min','max'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    save_table(grouped, '08_numericos_por_clase_train.csv', True)
    res = train.groupby(['ancho','alto']).size().sort_values(ascending=False).rename('frecuencia').reset_index()
    save_table(res, '09_resoluciones_train.csv')
    bounds, flagged = [], []
    for col in COLS:
        q1,q3 = train[col].quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        low,high=train[col].lt(lo),train[col].gt(hi)
        bounds.append(dict(atributo=col,q1=q1,q3=q3,iqr=iqr,limite_inferior=lo,limite_superior=hi,inferiores=int(low.sum()),superiores=int(high.sum())))
        for side, mask in [('inferior',low),('superior',high)]:
            sub=train.loc[mask,['clase','archivo',col]].rename(columns={col:'valor'}).copy()
            sub['atributo']=col; sub['tipo']=side; flagged.append(sub)
    save_table(pd.DataFrame(bounds), '10_iqr_train.csv')
    save_table(pd.concat(flagged, ignore_index=True), '11_outliers_train.csv')
    save_table(train[COLS].corr(), '12_correlaciones_train.csv', True)
    # All plots are based on train only (except tabulated split counts).
    counts=table['train']
    ax=counts.plot.bar(rot=0, figsize=(8,4)); ax.bar_label(ax.containers[0]); ax.set(ylabel='Imagenes',title='Distribucion de clases - train'); save_fig('fig01_distribucion.png')
    for col in ['megapixeles','aspecto','brillo','contraste']:
        plt.figure(figsize=(8,4)); train.boxplot(column=col,by='clase',grid=False); plt.suptitle(''); plt.title(f'{col} por clase - train'); save_fig(f'fig02_box_{col}.png'); plt.close('all')
        plt.figure(figsize=(8,4)); plt.hist(train[col],bins=35); plt.xlabel(col); plt.ylabel('Frecuencia'); plt.title(f'{col} - train'); save_fig(f'fig03_hist_{col}.png')
        plt.figure(figsize=(8,4)); bins=np.histogram_bin_edges(train[col],bins=30)
        for c in CLASSES:
            arr=train.loc[train.clase.eq(c),col]
            if len(arr): plt.hist(arr,bins=bins,density=True,histtype='step',linewidth=1.5,label=c)
        plt.legend(); plt.xlabel(col); plt.ylabel('Densidad'); plt.title(f'{col} por clase - train'); save_fig(f'fig04_hist_clases_{col}.png')
    for x,y in [('ancho','alto'),('brillo','contraste')]:
        plt.figure(figsize=(8,5))
        for c in CLASSES:
            g=train.loc[train.clase.eq(c)]; plt.scatter(g[x],g[y],s=8,alpha=.35,label=c)
        plt.xlabel(x); plt.ylabel(y); plt.legend(); plt.title(f'{x} vs. {y} - train'); save_fig(f'fig05_disp_{x}_{y}.png')
    sample=pd.concat([g.sample(min(50,len(g)),random_state=SEED) for _,g in train.groupby('clase')])
    axes_cols=['megapixeles','aspecto','brillo','contraste']
    scaled=sample[axes_cols].copy(); spans=(train[axes_cols].max()-train[axes_cols].min()).replace(0,1)
    scaled=(scaled-train[axes_cols].min())/spans; scaled['clase']=sample.clase
    plt.figure(figsize=(9,5)); parallel_coordinates(scaled,'clase',alpha=.12); plt.ylabel('Escala min-max de train'); plt.title('Coordenadas paralelas: hasta 50 imagenes por clase'); save_fig('fig06_paralelas.png')
    save_table(sample[['clase','archivo']+axes_cols], '13_muestra_paralelas.csv')
    fig, axes=plt.subplots(len(CLASSES),4,figsize=(10,12),squeeze=False)
    sample_ids=[]
    for i,c in enumerate(CLASSES):
        g=train.loc[train.clase.eq(c)]; sel=g.sample(min(4,len(g)),random_state=SEED)
        for j,ax in enumerate(axes[i]):
            ax.axis('off')
            if j>=len(sel): continue
            r=sel.iloc[j]; path=ROOT/'train'/c/r.archivo
            with Image.open(path) as img: ax.imshow(img.convert('RGB'))
            ax.set_title(c.upper()); sample_ids.append(dict(clase=c,archivo=r.archivo))
    save_fig('fig07_muestras.png'); save_table(pd.DataFrame(sample_ids),'14_muestra_visual.csv')
    bundle=Path('evidencias_eda_yuca.zip')
    previous=['inventario.csv','duplicados_exactos.csv','casi_duplicados.csv','excluidas.csv','particiones_limpias.csv','casi_duplicados_muestra.png']
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*')):
            if p.is_file(): z.write(p,p.relative_to(RESULTS))
        for name in previous:
            p=RESULTS/name
            if p.exists(): z.write(p,Path('previos')/name)
        z.write(Path(__file__), 'eda_yuca_completo.py')
    print('\nDistribucion real:\n',table)
    print('\nCalidad:\n',json.dumps(quality,indent=2,ensure_ascii=False))
    print('\nArchivo para compartir:',bundle.resolve())

if __name__=='__main__': main()
