# Backlog: Historias de Usuario (Proyecto MUNL)

Este documento ha sido generado mediante la abstracción funcional de los requerimientos y próximos pasos técnicos expuestos en `DOCUMENTACION_FUNCIONAL.md`. Sigue estrictamente la metodología dictada en el workflow de análisis funcional.

---

## Épica 1: Sistema Autónomo de Evaluación y Ranking (ResNet18 + CIFAR10)

### HU-004: Motor de Validación de Artefactos y Tolerancia a Fallos
**User Story:**  
Como **Investigador Operativo**, quiero que **el script inspeccione automáticamente la ruta de artefactos y filtre métodos incompletos** para poder **evaluar los modelos existentes sin que un experimento fallido rompa toda la generación de reportes.**

**Dependencias:**
- **Bloqueado por:** Ejecución del pipeline principal (Steps 5 y 10 finalizados localmente en el servidor generando los resnet18/cifar10).
- **Bloqueante para:** HU-005 y HU-006 (no se puede evaluar si no hay una lista segura de inputs).

**Criterios de Aceptación (AC):**
- [x] El script utiliza `argparse` para apuntar a `artifacts-root`.
- [x] Aborta la ejecución entera de forma crítica si falta un baseline `naive` para alguna seed.
- [x] Si un subdirectorio de método no posee sus 10 combinaciones obligatorias (`.pth` + `_meta.txt`), el script activa un `Warning`, lo purga de la lista de evaluación válida y lo documenta categóricamente en la sección `failed_methods` del Manifest.

---

### HU-005: Caché y Base de Referencia Naive
**User Story:**  
Como **Analista de Datos**, quiero que **el sistema evalúe el modelo baseline 'naive' una única vez guardándolo en disco** para poder **ahorrar drásticos tiempos de GPU en futuras ejecuciones iterativas sobre la misma distribución.**

**Dependencias:**
- **Bloqueado por:** HU-004 (necesita validar existencia de todos los `.pth` naive).
- **Bloqueante para:** HU-006 (se necesita el baseline inferido para cruzar y restar los valores y proponer componentes de ranking relativos).

**Criterios de Aceptación (AC):**
- [x] El motor invoca transaccionalmente `ModelEvaluationFromPathApp` sobre las 10 seeds de `naive` contra los dataloaders base.
- [x] Extrae y almacena la evaluación en `naive_baseline_cache.csv`.
- [x] En lanzamientos posteriores, el script detecta dicho _caché_, lo parsea limpiamente con Pandas y omite la fase inferencial de carga en CUDA.

---

### HU-006: Motor de Ranking L2 y Generación Integral de Reportes
**User Story:**  
Como **Interesado en la Investigación (Stakeholder)**, quiero que **todos los métodos desaprendidos exitosos sean evaluados bajo una lógica asimétrica ponderada igual a la fase de HP-Search** para obtener **las métricas brutas y tabulaciones finales listas para exportar a un paper.**

**Dependencias:**
- **Bloqueado por:** HU-004 (Métodos sanos) y HU-005 (Baseline de restas).
- **Bloqueante para:** Ninguno. (Fase terminal del flujo lógico de tablas).

**Criterios de Aceptación (AC):**
- [x] El sistema cruza cada fila de inferencia evaluada con su seed homónima en la matriz `naive`.
- [x] Se ejecuta la función `indiscernibilidad()` del core para procesar el ratio MIA.
- [x] El `objective10_score` se consolida bajo la ecuación L2 norm ponderada analítica: `((1/3)*loss_comp)**2`... 
- [x] Se consolidan y emiten las 4 tablas analíticas (`raw_metrics`, `ranking_seed_level`, `method_summary` y `ranking_final`) en formatos duales `.csv` y `.md` en la ruta `/reports/`.
- [x] Todo se firma generando un `run_manifest.json` para auditoría temporal y estructural (parámetros de runtime guardados).

---

### HU-007: Generación de Tablas Formato Paper (DeepUnlearn arXiv:2410.01276)
**User Story:**  
Como **Interesado en la Investigación (Stakeholder)**, quiero **un script que recree la estructura de las tablas mostradas en el benchmark de 'Deep Unlearn' (arXiv:2410.01276), limitadas específicamente a la arquitectura ResNet18 sobre el dataset CIFAR-10**, para **obtener resultados en un formato dual (CSV para legibilidad y `.tex` para compilación directa con `\input{}` en el manuscrito LaTeX).**

**Dependencias:**
- **Bloqueado por:** HU-006 (Necesita las métricas *raw* ya logeadas por el motor de evaluación para hacer los cruces).
- **Bloqueante para:** Ninguno. 

**Criterios de Aceptación (AC):**
- [x] Crear (o extender) un script (ej. `generate_resnet18_cifar10_tables.py`) que parsee los logs de resultados existentes en `/reports/`.
- [x] Generar salidas duales (CSV y `.tex`) equivalentes a la **Tabla 1 del paper**: *Main Results*, evaluando los métodos con base en *Retention Deviation* (RetDev) e *Indiscernibility*.
- [x] Generar salidas duales equivalentes a la **Tabla 2 del paper**: *Run-time efficiency*, mostrando la eficiencia en tiempo de ejecución (RTE) de cada método vs. Retrain.
- [x] Generar salidas duales equivalentes a la **Tabla 6 del paper**: *Per dataset results*, con la granularidad exhaustiva requerida por el benchmark (RA, FA, TA, RR, FR, TR, RetDev, Indisc, T-MIA, RTE) únicamente para ResNet18+CIFAR10.
- [x] Generar salidas duales equivalentes a la **Tabla 13 del paper**: *Per Architectures Rankings*, agrupando el *ranking* de ResNet18.
- [x] Todas las tablas deben estructurarse en un directorio de salida como `/reports/paper_tables/`. Los ficheros `.tex` deben usar utilidades de pandas (como `to_latex`) con estilos formales (e.g. `booktabs`) sin índice numérico de filas, listos para importarse al documento maestro.
