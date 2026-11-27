"""Apoyo de implementación para el Proyecto Yuca (revisar y explicar antes de usar).

No contiene resultados de entrenamiento. Las hipótesis y el protocolo provienen
 del EDA del equipo. Los parámetros no especificados se documentan en README.
Ejecutar desde la raíz del repositorio. El test solo se abre con --evaluar.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode

ROOT = Path('data/cassavaleafdata')
MANIFEST = Path('results/particiones_limpias.csv')
OUT = Path('results/experimentos')
CLASSES = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
LABELS = list(range(len(CLASSES)))
EXPECTED = {'train': [455, 1430, 763, 2655, 316],
            'validation': [151, 478, 257, 884, 105],
            'test': [154, 476, 253, 885, 105]}
EXPECTED_HASH = '885fce54f03e0a5d227baf7ea0d8b5a580b377670f3677f090dbc7c0eaeb141a'
SEED, BATCH, EPOCHS, PATIENCE = 42, 32, 15, 3
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXPERIMENTS = ['E0', 'E1', 'E2', 'E3', 'ADICIONAL']


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def device_for_run():
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_manifest():
    sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if sha != EXPECTED_HASH:
        raise ValueError(f'SHA-256 distinto del EDA: {sha}. Revisar manifiesto antes de entrenar.')
    df = pd.read_csv(MANIFEST, dtype=str)
    if list(df.columns) != ['split', 'clase', 'archivo']:
        raise ValueError(f'Columnas inesperadas: {list(df.columns)}')
    if df.isna().any().any() or df.duplicated().any():
        raise ValueError('Manifiesto con filas duplicadas o faltantes.')
    if set(df['split']) != set(EXPECTED) or set(df['clase']) != set(CLASSES):
        raise ValueError('Clases o particiones inesperadas.')
    counts = pd.crosstab(df['split'], df['clase']).reindex(index=list(EXPECTED), columns=CLASSES)
    for split, expected in EXPECTED.items():
        if list(counts.loc[split]) != expected:
            raise ValueError(f'Conteos diferentes para {split}: {counts.loc[split].tolist()}')
    return df, sha


class CassavaDataset(Dataset):
    def __init__(self, df, split, transform):
        self.df = df.loc[df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = ROOT / row['split'] / row['clase'] / row['archivo']
        with Image.open(path) as image:
            x = self.transform(image.convert('RGB'))
        return x, CLASSES.index(row['clase']), row['archivo']


def make_transform(exp, training):
    # Lado corto y recorte proporcional: E0 escala el esquema 256/224 a 73/64.
    side, size = (73, 64) if exp == 'E0' else (256, 224)
    ops = [transforms.Resize(side, interpolation=InterpolationMode.BILINEAR, antialias=True)]
    if training and exp == 'ADICIONAL':
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR,
                                          fill=(124, 116, 104))]
    ops += [transforms.CenterCrop(size)]
    if training and exp == 'ADICIONAL':
        ops += [transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1.0, 1.0),
                                            interpolation=InterpolationMode.BILINEAR, antialias=True)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def loader(df, split, exp, batch=BATCH):
    dataset = CassavaDataset(df, split, make_transform(exp, split == 'train'))
    # Un proceso para máxima compatibilidad con macOS/MPS.
    return DataLoader(dataset, batch_size=batch, shuffle=split == 'train', num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.classifier = nn.Linear(64 * 2 * 2, len(CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(exp, pretrained):
    if exp == 'E0':
        return SmallCNN()
    if exp == 'E3':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.requires_grad_(False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.requires_grad_(False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
        if exp == 'E2':
            # Ultimo bloque residual invertido, no la proyeccion final 1x1.
            model.features[-2].requires_grad_(True)
    return model


def train_mode(model, exp):
    if exp == 'E0':
        model.train()
    else:
        # Evita modificar BatchNorm del backbone congelado (incluidas sus estadísticas).
        model.eval()
        if exp == 'E3':
            model.fc.train()
        else:
            model.classifier.train()
            if exp == 'E2':
                model.features[-2].train()


def optimizer_for(model, exp):
    if exp == 'E2':
        return torch.optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': 1e-3},
            {'params': model.features[-2].parameters(), 'lr': 1e-4}], weight_decay=0.01)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3, weight_decay=0.01)


def count_params(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'entrenables': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def class_weights(df, device):
    n = df[df['split'] == 'train']['clase'].value_counts().reindex(CLASSES).to_numpy()
    return torch.tensor(n.sum() / (len(CLASSES) * n), dtype=torch.float32, device=device)


def run_epoch(model, batches, device, criterion, exp, optimizer=None, collect=False):
    training = optimizer is not None
    if training:
        train_mode(model, exp)
    else:
        model.eval()
    true, pred, files, confidence = [], [], [], []
    loss_num = loss_den = forward_seconds = 0.0
    with torch.set_grad_enabled(training):
        for x, y, names in batches:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if collect:
                sync(device)
                start = time.perf_counter()
            logits = model(x)
            if collect:
                sync(device)
                forward_seconds += time.perf_counter() - start
            loss = criterion(logits, y)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Pérdida no finita; revisar el entrenamiento.')
            if training:
                loss.backward()
                optimizer.step()
            # CrossEntropyLoss ponderada con reduction=mean divide por la suma de pesos.
            weight = criterion.weight[y].sum().item()
            loss_num += loss.item() * weight
            loss_den += weight
            true.extend(y.detach().cpu().tolist())
            pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if collect:
                files.extend(names)
                confidence.extend(logits.detach().softmax(dim=1).max(dim=1).values.cpu().tolist())
    metrics = {'loss': loss_num / loss_den,
               'accuracy': float(accuracy_score(true, pred)),
               'macro_f1': float(f1_score(true, pred, labels=LABELS, average='macro', zero_division=0))}
    if collect:
        return metrics, true, pred, files, confidence, forward_seconds
    return metrics


def curves(history, path):
    df = pd.DataFrame(history)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, field, label in [(axs[0], 'loss', 'Pérdida'), (axs[1], 'macro_f1', 'Macro-F1')]:
        ax.plot(df['epoch'], df['train_'+field], marker='o', label='Train')
        ax.plot(df['epoch'], df['val_'+field], marker='o', label='Validation')
        ax.set_xlabel('Época'); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def check(df, sha, device):
    print('Manifiesto SHA-256:', sha)
    print(pd.crosstab(df['clase'], df['split']))
    missing = [str(ROOT / r.split / r.clase / r.archivo) for r in df.itertuples()
               if not (ROOT / r.split / r.clase / r.archivo).is_file()]
    print('Archivos inexistentes:', len(missing))
    if missing:
        raise FileNotFoundError(missing[:5])
    print('Pesos de clase:', dict(zip(CLASSES, class_weights(df, torch.device('cpu')).tolist())))
    for exp in ['E0', 'E1', 'E3']:
        # Solo una muestra de train, pesos aleatorios para comprobar dimensiones, sin entrenar.
        x, y, _ = next(iter(loader(df, 'train', exp, batch=2)))
        model = build_model(exp, pretrained=False).to(device).eval()
        with torch.no_grad():
            result = model(x.to(device))
        assert tuple(result.shape) == (2, 5) and torch.isfinite(result).all().item()
        print(exp, 'entrada', list(x.shape), 'salida', list(result.shape), count_params(model))
        del model
    print('Comprobación correcta. No se entrenaron modelos ni se evaluó test.')


def train(df, sha, device, exp):
    folder = OUT / exp
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'{folder} ya contiene una corrida. No se sobrescribirá.')
    source = OUT / 'E1' / 'mejor.pt'
    if exp == 'E2':
        if not (OUT / 'E1' / 'entrenamiento.json').exists():
            raise FileNotFoundError('E2 requiere E1 terminado.')
        e1cfg = json.loads((OUT / 'E1' / 'config.json').read_text())
        if e1cfg['manifest_sha256'] != sha:
            raise ValueError('E1 y E2 tienen manifiestos distintos.')
    model = build_model(exp, pretrained=exp in ('E1', 'E3', 'ADICIONAL'))
    if exp == 'E2':
        model.load_state_dict(torch.load(source, map_location='cpu', weights_only=True))
    model.to(device)
    weights = class_weights(df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optimizer_for(model, exp)
    train_batches = loader(df, 'train', exp)
    val_batches = loader(df, 'validation', exp)
    folder.mkdir(parents=True, exist_ok=True)
    config = {'experimento': exp, 'manifest_sha256': sha, 'semilla': SEED,
              'batch': BATCH, 'max_epocas': EPOCHS, 'paciencia': PATIENCE,
              'device': str(device), 'torch': str(torch.__version__), 'clases': CLASSES,
              'pesos': weights.cpu().tolist(), 'parametros': count_params(model),
              'resolucion': 64 if exp == 'E0' else 224,
              'lr': [g['lr'] for g in optimizer.param_groups], 'weight_decay': 0.01,
              'pesos_iniciales': ('E1/mejor.pt' if exp == 'E2' else 'aleatorios' if exp == 'E0'
                                  else 'IMAGENET1K_V1'),
              'bloque_E2': 'features[-2]' if exp == 'E2' else None,
              'augmentation': exp == 'ADICIONAL'}
    save_json(folder / 'config.json', config)
    print('Configuración:', json.dumps(config, ensure_ascii=False))
    best, best_epoch, stale, history = -1.0, 0, 0, []
    sync(device); started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        sync(device); t = time.perf_counter()
        tr = run_epoch(model, train_batches, device, criterion, exp, optimizer)
        va = run_epoch(model, val_batches, device, criterion, exp)
        sync(device)
        history.append({'epoch': epoch, **{'train_'+k: v for k, v in tr.items()},
                        **{'val_'+k: v for k, v in va.items()}, 'seconds': time.perf_counter()-t})
        if va['macro_f1'] > best:
            best, best_epoch, stale = va['macro_f1'], epoch, 0
            torch.save(model.state_dict(), folder / 'mejor.pt')
        else:
            stale += 1
        pd.DataFrame(history).to_csv(folder / 'historial.csv', index=False)
        curves(history, folder / 'curvas.png')
        print(f'{exp} época {epoch}/{EPOCHS}: train loss={tr["loss"]:.4f}; '
              f'val acc={va["accuracy"]:.4f}; val macro-F1={va["macro_f1"]:.4f}; '
              f'mejor época={best_epoch}', flush=True)
        if stale >= PATIENCE:
            print('Early stopping por macro-F1 de validación.')
            break
    sync(device)
    summary = {'experimento': exp, 'epocas_ejecutadas': len(history), 'mejor_epoca': best_epoch,
               'mejor_macro_f1_validation': best, 'segundos_entrenamiento_total': time.perf_counter()-started,
               'checkpoint_mb': (folder / 'mejor.pt').stat().st_size / 1e6, **count_params(model)}
    save_json(folder / 'entrenamiento.json', summary)
    print(json.dumps(summary, indent=2))
    print('Entrenamiento terminado. Test no consultado.')


def evaluate(df, sha, device, exp):
    folder = OUT / exp
    if not (folder / 'entrenamiento.json').exists():
        raise FileNotFoundError(f'{exp} no tiene evidencia de entrenamiento terminado.')
    if (folder / 'test_metricas.json').exists():
        raise FileExistsError(f'{exp} ya fue evaluado; se conserva el resultado original.')
    cfg = json.loads((folder / 'config.json').read_text())
    if cfg['manifest_sha256'] != sha or cfg['clases'] != CLASSES:
        raise ValueError('Configuración no coincide con el manifiesto actual.')
    model = build_model(exp, pretrained=False)
    model.load_state_dict(torch.load(folder / 'mejor.pt', map_location='cpu', weights_only=True))
    model.to(device).eval()
    batches = loader(df, 'test', exp)
    criterion = nn.CrossEntropyLoss(weight=class_weights(df, device))
    # Calentamiento fuera de la medición (sin consultar métricas).
    with torch.no_grad():
        warm = torch.zeros(1, 3, cfg['resolucion'], cfg['resolucion'], device=device)
        for _ in range(5):
            model(warm)
        sync(device)
    m, true, pred, names, conf, seconds = run_epoch(model, batches, device, criterion, exp, collect=True)
    report = classification_report(true, pred, labels=LABELS, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / 'metricas_por_clase.csv')
    cm = confusion_matrix(true, pred, labels=LABELS)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(folder / 'matriz_confusion.csv')
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(LABELS, CLASSES); ax.set_yticks(LABELS, CLASSES)
    ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta real')
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(folder / 'matriz_confusion.png', dpi=180); plt.close(fig)
    predictions = pd.DataFrame({'archivo': names, 'real': [CLASSES[i] for i in true],
                                'predicha': [CLASSES[i] for i in pred], 'confianza': conf})
    predictions.to_csv(folder / 'predicciones.csv', index=False)
    predictions[predictions.real != predictions.predicha].to_csv(folder / 'errores.csv', index=False)
    m.update({'experimento': exp, 'n_test': len(true), 'device': str(device),
              'batch_inferencia': BATCH, 'ms_forward_por_imagen_promedio': 1000*seconds/len(true),
              'nota_tiempo': 'Forward por lotes sincronizados; no incluye lectura ni transferencia. No es latencia batch=1.'})
    save_json(folder / 'test_metricas.json', m)
    print(json.dumps(m, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--verificar', action='store_true')
    options.add_argument('--entrenar', choices=EXPERIMENTS)
    options.add_argument('--evaluar', choices=EXPERIMENTS)
    args = parser.parse_args()
    seed_all()
    device = device_for_run()
    print('Dispositivo:', device, '| PyTorch:', torch.__version__, flush=True)
    df, sha = read_manifest()
    if args.verificar:
        check(df, sha, device)
    elif args.entrenar:
        train(df, sha, device, args.entrenar)
    else:
        evaluate(df, sha, device, args.evaluar)


if __name__ == '__main__':
    main()
