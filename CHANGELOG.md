# CHANGELOG — Actividad 5: Predicción de Precios de Vivienda

Todos los cambios significativos del proyecto se documentan en este archivo.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/).

---

## [1.3.0] — 2025-06-20

### Añadido
- Registro completo de ambos experimentos en MLflow (parámetros, métricas, artefactos)
- Prueba t de Student pareada sobre los 5 folds de CV para validación estadística
- Figura de feature importance del modelo XGBoost
- Serialización de modelos en `fuentes/` (pickle y JSON)

### Modificado
- `train.py`: incorpora bloque `mlflow.start_run()` para cada modelo
- Panel MLflow configurado en `http://localhost:5000`

---

## [1.2.0] — 2025-06-18

### Añadido
- Grid Search con validación cruzada 5-fold para Ridge Regression y XGBoost
- Módulo `datos_prep.py` con funciones reutilizables de limpieza
- Figura comparativa: valor real vs. predicho para ambos modelos

### Modificado
- Hiperparámetros de XGBoost ajustados tras resultados de Grid Search
  - `n_estimators`: 100 → 200
  - `max_depth`: 6 → 4 (menor riesgo de sobreajuste)
  - `learning_rate`: 0.1 (confirmado óptimo)

### Decisión técnica
- Se descartó LinearRegression básica en favor de Ridge Regression como baseline,
  dado que el dataset presenta multicolinealidad moderada entre `habitaciones_prom`
  y `dormitorios_prom` (correlación = 0.85).

---

## [1.1.0] — 2025-06-15

### Añadido
- Ingeniería de características: `ratio_dorm_hab`, `ingreso_edad`
- Visualizaciones exploratorias: distribución del precio, correlación, scatter
- Split train/test 80-20 con `random_state=42` para reproducibilidad
- StandardScaler para Ridge Regression

### Modificado
- Eliminación de outliers IQR con `factor=3.0` en 3 variables (habitaciones, dormitorios, ocupantes)
- Se redujeron los registros de 20,640 a ~19,800 (retención ≈ 95.9%)

---

## [1.0.0] — 2025-06-12

### Añadido
- Inicialización del repositorio en GitHub
- Estructura de carpetas: `datos/`, `fuentes/`, `mlruns/`
- Carga del dataset California Housing desde `sklearn.datasets`
- Exploración inicial: dimensiones, valores nulos, estadísticas descriptivas
- Archivo `README.md` con guía de reproducción
- Configuración inicial de MLflow (`mlruns/` local)

### Decisiones técnicas iniciales
- Dataset seleccionado: California Housing (20,640 registros, tarea de regresión)
  - Justificación: dataset público, sin licencia restrictiva, rico en variables numéricas
    y con target continuo bien distribuido
- Modelos seleccionados: Ridge Regression (baseline) y XGBoost (principal)
  - Justificación: contraste entre modelo lineal regularizado y ensemble de árboles,
    cubriendo el espectro de complejidad-interpretabilidad

---

*Mantenido por Kevin Espindola Bravo — AL02963812*
