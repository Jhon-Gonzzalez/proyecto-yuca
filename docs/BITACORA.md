# Bitácora técnica — Proyecto Yuca

**Integrantes:** Jhon Mario González Acosta y Fabián Andrés Salgado  
**Asignatura:** Aprendizaje Computacional — Ingeniería de Sistemas, Universidad de Córdoba  
**Repositorio:** https://github.com/Jhon-Gonzzalez/proyecto-yuca

## Registro verificable de actividades

Las fechas de los primeros cinco registros corresponden a los commits locales reportados por el equipo. Los registros posteriores corresponden a la sesión del 26 de noviembre de 2026, hora local de Colombia. Esta bitácora documenta evidencias técnicas; no atribuye tareas individuales que no hayan sido confirmadas.

| Fecha | Actividad | Evidencia |
|---|---|---|
| 18/11/2026 | Configuración de Python, PyTorch, entorno virtual y MPS | `20b810a`, `check_env.py`, `requirements.txt` |
| 18/11/2026 | Inventario de iCassava 2019: 9.430 imágenes originales | `df30018`, `results/inventario.csv` |
| 19/11/2026 | Detección de duplicados exactos y perceptuales; comparación de umbrales dHash 4 y 0 | `2ab6087`, `results/duplicados_exactos.csv`, `results/casi_duplicados.csv` |
| 19/11/2026 | Limpieza del manifiesto: 63 exclusiones, 9.367 imágenes restantes | `986dea6`, `results/particiones_limpias.csv`, `results/excluidas.csv` |
| 24/11/2026 | Ejecución del EDA: 9.367 archivos leídos, cero errores | Salida de terminal y ZIP de evidencias EDA |
| 25/11/2026 | Registro del análisis descriptivo, IQR, calidad y visualizaciones | `fd323c7`, `src/eda_yuca_completo.py`, `results/eda_completo/` |
| 26/11/2026 | Conexión del repositorio local con GitHub y publicación de los cinco commits originales | `git push -u origin main` exitoso |
| 26/11/2026 | Verificación del pipeline, GPU MPS, manifiesto y dimensiones de salida | `fc39816`, `src/entrenar_yuca.py`, `src/ejecutar_yuca.py`, `docs/protocolo_implementacion.txt` |
| 26/11/2026 | Entrenamiento de E0, E1, E2, E3 y ADICIONAL; selección del checkpoint por Macro-F1 de validación | `04f91f3`, `results/experimentos/` |
| 26/11/2026 | Evaluación de los cinco checkpoints sobre las 1.873 imágenes de Test | `test_metricas.json`, matrices, predicciones, `comparacion.csv`, ZIP de evidencias |
| 26/11/2026 | Generación de ocho ejemplos de error y cuatro mapas Grad-CAM de E2; concordancia CPU/MPS en los ocho casos | `8_errores_E2.csv`, `8_errores_E2.png`, `4_gradcam_E2.png` (evidencias locales) |

## Resultados registrados

| Experimento | Épocas | Mejor época | Macro-F1 validación | Accuracy test | Macro-F1 test |
|---|---:|---:|---:|---:|---:|
| E0 | 11 | 8 | 40,75 % | 57,88 % | 44,95 % |
| E1 | 6 | 3 | 67,63 % | 73,36 % | 66,33 % |
| E2 | 11 | 8 | 74,28 % | 79,98 % | 73,56 % |
| E3 | 6 | 3 | 63,66 % | 72,08 % | 63,37 % |
| ADICIONAL | 12 | 9 | 66,57 % | 70,53 % | 64,47 % |

## Incidencias técnicas y resolución

- Se identificaron duplicados y etiquetas contradictorias. Se registraron exclusiones en el manifiesto sin borrar las imágenes originales.
- La primera ejecución del EDA falló porque el script no estaba en Descargas; la ejecución posterior procesó correctamente todas las imágenes limpias.
- En la primera ejecución del análisis visual faltaban dos archivos en Descargas. Se distribuyeron en un ZIP y posteriormente se obtuvieron las tres evidencias solicitadas.
- No se reportaron errores durante los cinco entrenamientos ni en su evaluación final.

## Trazabilidad y límites

- Manifiesto SHA-256: `885fce54f03e0a5d227baf7ea0d8b5a580b377670f3677f090dbc7c0eaeb141a`.
- Semilla de entrenamiento: 42; una corrida por configuración. No se ha medido estabilidad entre semillas.
- Las imágenes originales y los checkpoints se conservan localmente. No publicar fotografías sin verificar la licencia.
- Los resultados de test no deben utilizarse para ajustar retrospectivamente el protocolo ni las hipótesis originales.

## Pendientes de validación por el equipo

- Registrar la distribución real de responsabilidades de los integrantes.
- Confirmar las decisiones de implementación y redactar las interpretaciones propias de los resultados.
- Completar las columnas de observación y causa hipotética de los ocho errores. Distinguir observación de causalidad demostrada.
- Demostración de inferencia individual: completada; ver cierre técnico.
- Commit de evaluación final: `18aa651`, publicado.
- Completar informe, conclusiones, declaración de ayuda de IA y preparación de sustentación.

## Uso de IA

Se utilizó ChatGPT como apoyo para la programación de scripts, organización de evidencias y documentación técnica. Las hipótesis, decisiones experimentales, interpretaciones de errores y conclusiones deben ser revisadas y desarrolladas por el equipo de acuerdo con las instrucciones del docente. El equipo deberá ajustar esta declaración para que refleje fielmente el uso realizado.

## Cierre técnico — 26 de noviembre de 2026 (hora local)

- Commit `04f91f3`: entrenamiento de las cinco configuraciones; evaluación de test ejecutada después de finalizar los entrenamientos.
- Incidencia: faltaban los archivos individuales en Descargas; se utilizó el ZIP de cierre visual y se verificó la generación de ocho errores y cuatro Grad-CAM.
- Incidencia: `git diff` abrió el paginador y faltó descargar la bitácora; se salió del visor y se repitió con `git --no-pager`.
- Commit `18aa651`: evaluación final, métricas, matrices, errores, README y bitácora. Push confirmado; rama main sincronizada y árbol limpio.
- Inferencia individual ejecutada: `test-cbb-95.jpg`, real CBB, predicha Healthy, softmax 0,9800334572792053; coincide con la predicción previamente registrada.
- Pendientes del equipo: responsabilidades reales; observación, hipótesis de causa y revisión de cada error; interpretación de métricas, conclusiones y declaración de ayudas; acreditar control intermedio según corresponda.

## Ampliación tras el control intermedio — 26 de noviembre de 2026 (hora Colombia)

- El integrante confirmó que ambos participaron en el proyecto y presentaron el control intermedio. Según su relato, el docente pidió más de tres modelos y análisis riguroso. No hay acta ni fecha exacta de la revisión; no se afirma aprobación formal ni cumplimiento del plazo.
- `df47ed7`: documentación original. `3d00d62`: registro de control y autoría. `b209b5b`: código de E5 y observaciones del docente.
- Problema: tres arquitecturas no satisfacían la nueva indicación comunicada. Alternativa considerada: EfficientNet-B0 como tercera arquitectura preentrenada. Implementación asistida: mismo régimen de E1/E3 (extractor congelado, 224 × 224, ImageNet V1, AdamW, semilla 42, batch 32, 15 épocas, paciencia 3, pérdida ponderada); comprobar y revisar por el equipo.
- Prueba de verificación MPS correcta: 4.013.953 parámetros, 6.405 entrenables; manifiesto íntegro. Primer intento falló al duplicar el comando (`E5caffeinate`); evaluación prematura rechazó la falta de entrenamiento. Se encadenaron entrenamiento, evaluación y exportación con `&&`.
- Ejecución E5: 13 épocas, mejor época 10, Macro-F1 validación 71,58 %, Accuracy validación 78,99 % (redondeado del log), 537,13 s, checkpoint 16,330 MB. Test: Accuracy 78,96 %, Macro-F1 71,80 %, 1.479 aciertos y 394 errores de 1.873; forward promedio 2,816 ms/imagen por lotes.
- Verificación posterior: 11.238 predicciones de seis experimentos coinciden con manifiesto, métricas y matrices de confusión. Los 50 archivos de los cinco experimentos originales son idénticos a los del ZIP anterior.
- E5 se realizó después del primer test; la comparación adicional es exploratoria. Se requiere validación independiente para confirmar el resultado y más semillas para evaluar estabilidad.
- Participación conjunta declarada de ambos integrantes; propuesta de exposición: Jhon (datos, limpieza, EDA, pipeline), Fabián (arquitecturas, métricas, errores), ambos (discusión y demostración). No se atribuyen retrospectivamente tareas individuales no comprobadas.
- Se actualizó el informe con la cuarta arquitectura y los seis experimentos. Las interpretaciones y conclusiones propias siguen siendo responsabilidad de ambos, según la sección 8 del enunciado. La ayuda de IA abarcó generación y revisión de scripts, análisis de cifras y redacción, y debe declararse íntegramente.
