# Proyecto Yuca

Comparación experimental de modelos para clasificar fotografías de hojas de yuca (iCassava 2019).

**Universidad de Córdoba — Ingeniería de Sistemas — Aprendizaje Computacional**  
Integrantes: Jhon Mario González Acosta y Fabián Andrés Salgado Mosquera.  
Docente: Oswaldo Enrique Vélez Langs.

## Datos y reproducibilidad

- Original: 9.430 imágenes. Manifiesto limpio: 9.367 (train 5.619, validation 1.875, test 1.873).
- Clases: CBB, CBSD, CGM, CMD y Healthy.
- SHA-256 del manifiesto: `885fce54f03e0a5d227baf7ea0d8b5a580b377670f3677f090dbc7c0eaeb141a`.
- Dataset original: [iCassava 2019](https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip). Colocar las carpetas train, validation y test bajo `data/cassavaleafdata/`.
- No se incluyen los archivos originales del dataset ni checkpoints. No redistribuir fotografías sin verificar las condiciones de uso. El repositorio contiene el historial de preparación de datos.
- Entorno registrado: Python 3.14.3, PyTorch 2.15.0, torchvision 0.29.2; Apple M4/MPS. Consultar `requirements.txt`.

## Ejecución

Desde la raíz, con el dataset disponible:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -u src/ejecutar_yuca.py verificar
```

En un directorio de resultados nuevo (el programa no sobrescribe corridas):

```bash
python -u src/ejecutar_yuca.py entrenar
python -u src/ejecutar_yuca.py evaluar
```

El segundo comando debe ejecutarse solo después de fijar el protocolo. Si ya existen resultados, preservarlos antes de repetir el experimento. E2 depende del checkpoint de E1. Los pesos ImageNet se descargan durante la primera ejecución de E1/E3/E5.

## Protocolo y experimentos

Semilla 42, batch 32, máximo 15 épocas, early stopping con paciencia 3 por macro-F1 de validación, AdamW, CrossEntropyLoss ponderada por clase. E0 usa 64×64; las demás configuraciones 224×224. Ver `docs/protocolo_implementacion.txt` y los JSON de cada corrida para los detalles.

- E0: CNN pequeña entrenada desde cero.
- E1: MobileNetV2 con extractor congelado.
- E2: MobileNetV2, ajuste parcial desde el mejor checkpoint de E1.
- E3: ResNet18 con extractor congelado.
- ADICIONAL (E4): E1 con aumentos geométricos de datos solo durante entrenamiento.
- E5: EfficientNet-B0 ImageNet V1, extractor congelado y cabeza entrenable. Ampliación tras el control intermedio, según lo informado por el equipo; protocolo de E1/E3, sin augmentation.

## Resultados de la corrida registrada

Las cifras corresponden a 1.873 imágenes de test por experimento, con checkpoints elegidos por macro-F1 de validación. Se verificaron contra los CSV de predicciones.

| Experimento | Macro-F1 val. | Accuracy test | Macro-F1 test | Época elegida | Tiempo (s) | Tamaño (MB) |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.4075 | 0.5788 | 0.4495 | 8 | 206.68 | 0.103 |
| E1 | 0.6763 | 0.7336 | 0.6633 | 3 | 203.19 | 9.145 |
| E2 | 0.7428 | 0.7998 | 0.7356 | 8 | 412.62 | 9.145 |
| E3 | 0.6366 | 0.7208 | 0.6337 | 3 | 242.18 | 44.787 |
| ADICIONAL | 0.6657 | 0.7053 | 0.6447 | 9 | 503.57 | 9.145 |
| E5 | 0.7158 | 0.7896 | 0.7180 | 10 | 537.13 | 16.330 |

Fuente: `results/experimentos/comparacion.csv` y los archivos `test_metricas.json`, `predicciones.csv`, `matriz_confusion.csv` e `historial.csv` de cada corrida.

El tiempo de inferencia registrado es el promedio por imagen de forwards sincronizados en lotes de 32, sin lectura ni transferencia; no equivale a latencia individual.

## Organización

- `src/`: inventario, duplicados, limpieza, EDA, entrenamiento y evaluación.
- `results/particiones_limpias.csv`: manifiesto de las imágenes autorizadas.
- `results/eda_completo/`: resultados exploratorios.
- `results/experimentos/`: configuraciones, curvas, métricas, matrices y predicciones.
- `docs/protocolo_implementacion.txt`: decisiones técnicas y declaración de asistencia.

## Alcance y limitaciones

Cuatro arquitecturas, seis experimentos. E5 se añadió después de evaluar los cinco originales y se evaluó en el mismo test; la comparación adicional es exploratoria. No se modificaron las hipótesis originales ni los resultados previos. Una corrida por configuración no permite estimar estabilidad entre semillas. No se dispone de identificadores de planta o finca, y no se ha certificado la ausencia de todos los casi duplicados. Las imágenes proceden de Uganda; el desempeño no constituye validación agronómica para Colombia.

## Referencias y asistencia

- Mwebaze et al. (2019). [iCassava 2019 Fine-Grained Visual Categorization Challenge](https://arxiv.org/abs/1908.02900).
- [TensorFlow Datasets: cassava](https://www.tensorflow.org/datasets/catalog/cassava).
- ChatGPT se usó para generar y revisar scripts, documentación, organización de evidencias y verificación de métricas. Su uso fue más amplio que consultas conceptuales y depuración puntual; debe declararse y revisarse frente a la sección 8 del enunciado. Los entrenamientos y la evaluación se ejecutaron localmente. Las decisiones e interpretaciones deben ser revisadas y defendidas por los integrantes del equipo.
