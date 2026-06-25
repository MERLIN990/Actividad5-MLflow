# Actividad 5 — Predicción de Precios de Vivienda con MLflow

**Materia:** Gestión de Proyectos de Inteligencia Artificial  
**Alumno:** Kevin Espindola Bravo | AL02963812  
**Docente:** Luis Ariel Vazquez Piña  
**Fecha:** Junio 2025

---

## Descripción del Proyecto

Este repositorio implementa un pipeline completo de Machine Learning para la predicción del precio mediano de viviendas en California, utilizando el dataset California Housing (scikit-learn). Se comparan dos modelos: **Ridge Regression** (baseline) y **XGBoost** (modelo principal), con Grid Search, validación cruzada k-fold y registro completo en MLflow.

---

## Estructura del Repositorio

```
Actividad5/
│── datos/
│   ├── datos_ini/           # Datos originales descargados
│   │   └── california_housing_raw.csv
│   └── datos_limp/          # Datos tras limpieza y feature engineering
│       └── california_housing_clean.csv
│── fuentes/
│   ├── entrena.ipynb        # Notebook de Colab (exploración + entrenamiento)
│   ├── datos_prep.py        # Funciones modulares de limpieza y preparación
│   ├── train.py             # Script principal: entrenamiento y registro MLflow
│   ├── ridge_model.pkl      # Modelo Ridge serializado (joblib)
│   ├── xgboost_model.json   # Modelo XGBoost serializado
│   ├── scaler.pkl           # Escalador StandardScaler
│   └── figuras/             # Visualizaciones generadas automáticamente
│── mlruns/                  # Directorio de experimentos MLflow (auto-generado)
│── README.md                # Este archivo
│── CHANGELOG.md             # Historial de cambios del proyecto
│── requirements.txt         # Dependencias Python
```

---

## Requisitos

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
scikit-learn>=1.3.0
xgboost>=2.0.0
mlflow>=2.10.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
joblib>=1.3.0
jupyter>=1.0.0
```

---

## Guía de Reproducción Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/kevin-espindola/actividad5-mlflow.git
cd actividad5-mlflow
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

### 3. Preparar los Datos

```bash
python fuentes/datos_prep.py
```
Genera `datos/datos_ini/california_housing_raw.csv` y `datos/datos_limp/california_housing_clean.csv`.

### 4. Entrenar Modelos y Registrar en MLflow

```bash
python fuentes/train.py
```
Al finalizar, los experimentos quedan registrados en la carpeta `mlruns/`.

### 5. Ver Resultados en el Panel de MLflow

```bash
mlflow ui
```
Abrir `http://localhost:5000` en el navegador. Se visualizan parámetros, métricas y artefactos de ambos modelos.

### 6. Ejecutar el Notebook Interactivo (Opcional)

```bash
jupyter notebook fuentes/entrena.ipynb
```

---

## Dataset: California Housing

| Atributo | Descripción |
|---|---|
| **Fuente** | `sklearn.datasets.fetch_california_housing` |
| **Origen** | Censo de California 1990 (StatLib) |
| **Registros originales** | 20,640 |
| **Registros tras limpieza** | ~19,800 (tasa retención ≈ 95.9%) |
| **Features** | 10 (8 originales + 2 derivadas) |
| **Variable objetivo** | `precio_mediano` (×$100,000 USD) |

### Diccionario de Variables

| Variable | Tipo | Descripción |
|---|---|---|
| `ingreso_mediano` | Numérica | Ingreso mediano del bloque censal (×$10,000 USD) |
| `edad_casa` | Numérica | Mediana de años de antigüedad de las viviendas |
| `habitaciones_prom` | Numérica | Promedio de habitaciones por hogar |
| `dormitorios_prom` | Numérica | Promedio de dormitorios por hogar |
| `poblacion` | Numérica | Población total del bloque censal |
| `ocupantes_prom` | Numérica | Promedio de ocupantes por vivienda |
| `latitud` | Numérica | Latitud geográfica del bloque |
| `longitud` | Numérica | Longitud geográfica del bloque |
| `ratio_dorm_hab` | Derivada | Proporción dormitorios / habitaciones |
| `ingreso_edad` | Derivada | Interacción ingreso mediano × edad de la casa |
| `precio_mediano` | **Objetivo** | Precio mediano de vivienda en el bloque (×$100,000 USD) |

---

## Modelos y Resultados

| Métrica | Ridge Regression | XGBoost | Ganador |
|---|---|---|---|
| MAE | ~0.49 | ~0.31 | XGBoost |
| RMSE | ~0.72 | ~0.46 | XGBoost |
| R² | ~0.62 | ~0.84 | XGBoost |
| MAPE | ~25.1% | ~15.4% | XGBoost |
| CV R² (5-fold) | ~0.61 ± 0.01 | ~0.84 ± 0.01 | XGBoost |

La diferencia entre modelos es estadísticamente significativa (prueba t pareada, p < 0.05).

---

## Decisiones Técnicas

- **Ridge Regression como baseline:** Modelo lineal regularizado, adecuado para detectar multicolinealidad entre features. Fácil de interpretar y reproducir.
- **XGBoost como modelo principal:** Captura relaciones no lineales complejas entre variables. Resistente a outliers residuales. Mejor rendimiento generalizable.
- **Grid Search + 5-fold CV:** Búsqueda exhaustiva de hiperparámetros con evaluación robusta para evitar sobreajuste.
- **StandardScaler solo para Ridge:** XGBoost es invariante al escalado de features; Ridge sí lo requiere.
- **Feature engineering:** `ratio_dorm_hab` e `ingreso_edad` aumentan el poder predictivo en ~2 pp de R².

---

## Deuda Técnica y MLOps

- [ ] Implementar monitoreo de data drift con Evidently AI
- [ ] Pipeline de reentrenamiento mensual automático
- [ ] Explorar LightGBM y CatBoost como alternativas a XGBoost
- [ ] Contenedorización del pipeline con Docker
- [ ] Despliegue como API REST con FastAPI + MLflow Model Registry

---

## Autor

**Kevin Espindola Bravo**  
Matrícula: AL02963812  
Máster en Inteligencia Artificial — Universidad Tecmilenio
