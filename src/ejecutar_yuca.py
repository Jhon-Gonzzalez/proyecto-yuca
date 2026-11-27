"""Orquesta las corridas existentes y exporta solo evidencias reales."""
import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

EXPS = ('E0', 'E1', 'E2', 'E3', 'ADICIONAL')
OUT = Path('results/experimentos')
TRAINER = Path('src/entrenar_yuca.py')


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, '-u', str(TRAINER), *args], check=True)


def completed(exp):
    folder = OUT / exp
    return all((folder / f).is_file() for f in ('entrenamiento.json', 'mejor.pt', 'config.json', 'historial.csv'))


def train_all():
    run('--verificar')
    for exp in EXPS:
        if completed(exp):
            print(exp, 'ya terminado; no se sobrescribe.', flush=True)
        else:
            run('--entrenar', exp)
    print('\nEntrenamientos completos. Test no consultado por este comando.')


def evaluate_all():
    missing = [exp for exp in EXPS if not completed(exp)]
    if missing:
        raise SystemExit('Faltan entrenamientos completos: ' + ', '.join(missing))
    for exp in EXPS:
        if (OUT / exp / 'test_metricas.json').exists():
            print(exp, 'ya evaluado; se conserva el resultado.', flush=True)
        else:
            run('--evaluar', exp)


def export():
    rows = []
    for exp in EXPS:
        folder = OUT / exp
        if not (folder / 'entrenamiento.json').is_file():
            continue
        train = json.loads((folder / 'entrenamiento.json').read_text())
        test_path = folder / 'test_metricas.json'
        test = json.loads(test_path.read_text()) if test_path.is_file() else {}
        rows.append({'experimento': exp, 'estado': 'evaluado' if test else 'entrenado',
                     'epocas': train['epocas_ejecutadas'], 'mejor_epoca': train['mejor_epoca'],
                     'macro_f1_validation': train['mejor_macro_f1_validation'],
                     'accuracy_test': test.get('accuracy', ''), 'macro_f1_test': test.get('macro_f1', ''),
                     'n_test': test.get('n_test', ''), 'segundos_entrenamiento': train['segundos_entrenamiento_total'],
                     'ms_forward_por_imagen': test.get('ms_forward_por_imagen_promedio', ''),
                     'parametros_total': train['total'], 'parametros_entrenables': train['entrenables'],
                     'checkpoint_mb': train['checkpoint_mb']})
    if not rows:
        raise SystemExit('No hay entrenamientos terminados para exportar.')
    table = OUT / 'comparacion.csv'
    with table.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    history = subprocess.run(['git','log','--oneline','--date=iso'], capture_output=True, text=True)
    (OUT / 'git_historial.txt').write_text(history.stdout, encoding='utf-8')
    target = Path('evidencias_experimentos_yuca.zip')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(OUT.rglob('*')):
            if file.is_file() and file.suffix.lower() in {'.csv', '.json', '.png', '.txt', '.log'}:
                archive.write(file, file.as_posix())
        for file in (TRAINER, Path(__file__), Path('requirements.txt'), Path('results/particiones_limpias.csv')):
            if file.is_file():
                archive.write(file, file.as_posix())
    print('Comparacion:', table.resolve())
    print('Evidencias:', target.resolve())
    print('No se incluyen pesos ni fotografias originales. Las metricas ausentes quedan vacias.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion', choices=['verificar', 'entrenar', 'evaluar', 'exportar'])
    args = parser.parse_args()
    if not TRAINER.is_file():
        raise SystemExit('Ejecutar desde la raiz del proyecto con src/entrenar_yuca.py instalado.')
    if args.accion == 'verificar':
        run('--verificar')
    elif args.accion == 'entrenar':
        train_all(); export()
    elif args.accion == 'evaluar':
        evaluate_all(); export()
    else:
        export()


if __name__ == '__main__':
    main()
