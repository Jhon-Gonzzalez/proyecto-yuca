# Guía de sustentación — Proyecto Yuca
Universidad de Córdoba · Aprendizaje Computacional · 2026

## Secuencia sugerida de exposición (8–10 minutos)
1. Problema y alcance (1 min). Clasificación supervisada en cinco clases; comparación académica, no diagnóstico agronómico.
2. Datos y control de calidad (1 min). iCassava 2019, 9.430 imágenes originales; 63 exclusiones; manifiesto limpio de 9.367 (5.619/1.875/1.873). MD5 y dHash; particiones independientes según los controles realizados; falta ID de planta/finca.
3. EDA y protocolo (1 min). CMD 2.655 frente a Healthy 316 en train; pesos de clases; Macro-F1; semilla 42, AdamW, lotes de 32, máximo 15 épocas y paciencia 3. E0 64×64, demás 224×224.
4. Arquitecturas y experimentos (1–2 min). E0 CNN propia; E1 MobileNetV2 congelada; E2 ajuste parcial desde E1; E3 ResNet18 congelada; E4/ADICIONAL repetición de E1 con augmentation; E5 EfficientNet-B0 congelada. Distinguir arquitectura y régimen de entrenamiento.
5. Resultados (1–2 min). Mostrar `comparacion.csv` y curvas. Los seis experimentos usaron 1.873 imágenes de test. E2: 79,98 % accuracy, 73,56 % Macro-F1, 1.498 aciertos y 375 errores. E5: 78,96 % accuracy y 71,80 % Macro-F1. Comparar E1/E3, E1/E2 y E5 con los extractores congelados; discutir ustedes la relación con las hipótesis, recursos y estabilidad.
6. Errores y límites (1 min). Mostrar ocho imágenes y Grad-CAM localmente. Cada integrante explica sus propias observaciones, hipótesis causales y qué prueba haría para comprobarlas. Los scores no están calibrados; no afirmar causalidad a partir de Grad-CAM.
7. Demostración operativa y cierre (1–2 min). Cargar datos, mostrar una inferencia correcta y una incorrecta. Cada integrante explica lo que hizo y su propia conclusión.

## Demostración en terminal (desde la raíz)

```bash
source .venv/bin/activate
python check_env.py
python -u src/ejecutar_yuca.py verificar
git --no-pager log --oneline
```

Carga de una imagen con el Dataset utilizado por el pipeline:

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from entrenar_yuca import read_manifest, CassavaDataset, make_transform
manifest, sha = read_manifest()
dataset = CassavaDataset(manifest, 'test', make_transform('E2', False))
x, y, nombre = dataset[0]
print('Archivo:', nombre, 'Tensor:', tuple(x.shape), 'Etiqueta:', y)
print('Total test:', len(dataset))
PY
```

Inferencia de un acierto y un error conocidos:

```bash
python -u src/analizar_errores_yuca.py --imagen data/cassavaleafdata/test/cbb/test-cbb-141.jpg
python -u src/analizar_errores_yuca.py --imagen data/cassavaleafdata/test/cbb/test-cbb-95.jpg
```

Recalcular Macro-F1 sin volver a entrenar ni alterar test:

```bash
python - <<'PY'
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
p = pd.read_csv('results/experimentos/E2/predicciones.csv')
print('Accuracy:', accuracy_score(p.real, p.predicha))
print('Macro-F1:', f1_score(p.real, p.predicha, average='macro'))
PY
```

## Preguntas de práctica
| Pregunta | Concepto o evidencia que debes explicar |
|---|---|
| ¿Por qué es supervisado? | Etiquetas originales por carpeta y manifiesto CSV; no se generaron etiquetas clínicas nuevas. |
| ¿Cómo evitan la fuga de datos? | Control MD5/dHash, manifiesto limpio, test no usado para ajustar hiperparámetros; riesgo residual por capturas relacionadas. |
| ¿Qué hace la pérdida ponderada? | Penaliza los errores según pesos inversos a frecuencia de train; no inventa imágenes. |
| ¿Qué es Macro-F1? | Media aritmética del F1 de las cinco clases, igual peso por clase. |
| ¿Qué se congela en E1, E3 y E5? | Extractor de características; solo se ajusta cabeza. BatchNorm congelada en modo evaluación. |
| ¿Qué cambia en E2? | Carga E1, descongela features[-2], LR 0,0001 en bloque y 0,001 en cabeza. |
| ¿Cómo comparas E1 y E3? | Mismo régimen de extracción, mismos splits, presupuesto y transformaciones; comparar Macro-F1 y recursos. |
| ¿Es E4 una arquitectura distinta? | No, es una configuración experimental de MobileNetV2. |
| ¿Qué diferencia E0 de E1? | Arquitectura, resolución y preentrenamiento; no se puede atribuir toda la diferencia a una sola variable. |
| ¿Qué observas en las curvas? | Revisar train/val loss y F1, especialmente épocas 8–11 de E2; formular interpretación propia. |
| ¿Por qué falló E4 respecto a la hipótesis? | Revisar las cifras y proponer una hipótesis comprobable; no confundir con una causa demostrada. |
| ¿Por qué puede fallar con score 0,98? | Softmax no calibrado; la etiqueta y la predicción pueden discrepar incluso con score alto. |
| ¿Qué aporta Grad-CAM? | Localización de contribución positiva para una clase, no demuestra la causa de la predicción. |
| ¿El tiempo de 2,01 ms es latencia de una foto? | No: forward medio por imagen en lotes de 32, sin lectura/transferencia. |
| ¿Es estable entre semillas? | No se ha medido: solo una corrida por configuración. |
| ¿Funcionará en Colombia? | No está demostrado; hacen falta datos locales compatibles y validación externa. |
| ¿Qué hizo cada integrante? | Cada uno debe explicar contribuciones reales, decisión, dificultad, prueba y resultado; completar bitácora. |

## Antes de entregar
- Completar observación y causa hipotética de los ocho casos y conclusiones con palabras del equipo.
- Confirmar roles y declaración de ayuda externa; el enunciado no permite delegar interpretación ni conclusiones a IA.
- Acreditar, si corresponde, el control intermedio obligatorio; no se puede sustituir retrospectivamente por el informe.
- Revisar permiso de redistribución del dataset antes de publicar imágenes; conservar las figuras fotográficas localmente.

## Ampliación E5
- Cuatro arquitecturas distintas, seis experimentos. E5 usa el mismo protocolo que E1/E3. Tiene 4.013.953 parámetros totales y 6.405 entrenables; 13 épocas, mejor época 10.
- Mostrar `results/experimentos/E5/curvas.png`, `matriz_confusion.csv` y `metricas_por_clase.csv`. E5 supera E1 en 5,47 pp de Macro-F1 test; queda 1,76 pp por debajo de E2.
- Preguntas para responder con razonamiento propio: ¿por qué comparar primero E1/E3/E5? ¿Qué implica que E5 tenga recall Healthy superior a E2 y recall CBB inferior? ¿Qué limitación introduce añadir E5 después del primer test?
- Propuesta de exposición: Jhon explica datos, limpieza, EDA y carga; Fabián arquitecturas, métricas, errores; ambos explican decisiones y demostración.
