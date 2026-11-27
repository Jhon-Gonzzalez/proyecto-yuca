# Cierre técnico y verificación del control intermedio

Fecha de documentación: 26 de noviembre de 2026 (Colombia). Registro preparado con asistencia de ChatGPT. No constituye acta del docente ni acredita asistencia.

## Evidencias verificables
| Requisito del enunciado (sección 6) | Evidencia disponible | Estado |
|---|---|---|
| Repositorio con historial de días distintos | Commits 20b810a, df30018 (18/11); 2ab6087, 986dea6 (19/11); fd323c7 (25/11); fc39816, 04f91f3, 18aa651 y df47ed7 (26/11, hora local) | Disponible |
| E0 ejecutado y E1 al menos parcial | results/experimentos/E0 e E1; ambos entrenados el 26/11 | Disponible al cierre; no acredita plazo del día 7 |
| Métricas iniciales y registro de errores | historial.csv, predicciones.csv, errores.csv y docs/8_errores_E2.csv | Disponible al cierre |
| Carga de datos e inferencia | Verificación del manifiesto de 9.367 imágenes y ejecución individual de test-cbb-95.jpg el 26/11; predicción healthy, score 0,9800334572792053 | Ejecutadas localmente; demostración ante docente no acreditada |
| Modificación pequeña solicitada por el docente | No se ha aportado evidencia de solicitud ni ejecución ante docente | Por confirmar |
| Explicación individual y revisión del día 7 | No se ha aportado fecha, acta o confirmación del docente | Por confirmar |

El enunciado indica que la entrega final no sustituye retrospectivamente el control obligatorio. Los entrenamientos del 26/11 no deben presentarse como realizados en una fecha anterior. Si hubo una revisión o justificación aceptada, registrar fecha real, actividades demostradas y resultado.

## Comprobación de entrega
Tres arquitecturas (CNN pequeña, MobileNetV2 y ResNet18), cinco experimentos evaluados, 1.873 imágenes de test por experimento. Resultados en results/experimentos/comparacion.csv. Las evidencias fotográficas y checkpoints permanecen locales mientras no se comprueben permisos de redistribución. Informe sin anexos fotográficos publicado en df47ed7.

## Aportaciones y autoría
- Jhon Mario González Acosta: la sesión de terminal y los commits aportados documentan ejecución y publicación del código y resultados desde su cuenta. No permiten atribuirle en exclusiva el diseño ni la autoría del código.
- Fabián Andrés Salgado Mosquera: integrante identificado en el informe; su contribución individual aún no está documentada.
- ChatGPT: se utilizó para generar y revisar scripts de EDA, entrenamiento, evaluación, análisis visual y documentación. Esta ayuda va más allá de una simple consulta conceptual; se debe declarar y contrastar con la sección 8 del enunciado.
- El equipo debe consignar sus aportes, decisiones, explicación de errores y conclusiones propias. No se atribuyen declaraciones o aprobaciones no recibidas.

## Confirmaciones del equipo (pendientes)
1. Fecha y resultado del control intermedio; qué se ejecutó y qué indicó el docente.
2. Para cada integrante: implementación o tarea realizada, decisión tomada, dificultad resuelta y evidencia.
3. Interpretaciones propias de las comparaciones E1/E3 y E1/E2, augmentation, ocho errores y limitaciones. Ver informe, métricas y guía de sustentación.
