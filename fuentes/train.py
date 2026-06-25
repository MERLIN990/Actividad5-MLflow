"""
train.py — Actividad 5: Entrenamiento, ajuste y registro en MLflow
Proyecto: Predicción de Precios de Vivienda (Regresión)
Alumno: Kevin Espindola Bravo | Matrícula: AL02963812
Materia: Gestión de Proyectos de Inteligencia Artificial
"""

# ─────────────────────────────────────────────────────────
# INSTALACIÓN DE DEPENDENCIAS (Google Colab / entorno nuevo)
# ─────────────────────────────────────────────────────────
import subprocess, sys

for pkg in ["mlflow", "xgboost", "seaborn", "scipy", "joblib"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✓ Dependencias instaladas correctamente")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy import stats
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

warnings.filterwarnings("ignore")

import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN DE MLFLOW
# ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "mlruns"          # carpeta local
EXPERIMENT_NAME = "Prediccion_Precios_Vivienda"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ─────────────────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────
print("=" * 60)
print(" PASO 1 — Carga y exploración del dataset")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Renombrar columnas al español para mayor claridad
col_rename = {
    "MedInc":       "ingreso_mediano",
    "HouseAge":     "edad_casa",
    "AveRooms":     "habitaciones_prom",
    "AveBedrms":    "dormitorios_prom",
    "Population":   "poblacion",
    "AveOccup":     "ocupantes_prom",
    "Latitude":     "latitud",
    "Longitude":    "longitud",
    "MedHouseVal":  "precio_mediano"
}
df.rename(columns=col_rename, inplace=True)

print(f"\n► Dimensiones originales: {df.shape}")
print(f"► Valores nulos:\n{df.isnull().sum()}")
print(f"\n► Estadísticas descriptivas:\n{df.describe().round(3)}")

# Guardar datos originales
os.makedirs("datos/datos_ini", exist_ok=True)
os.makedirs("datos/datos_limp", exist_ok=True)
df.to_csv("datos/datos_ini/california_housing_raw.csv", index=False)
print("\n✓ Datos originales guardados en datos/datos_ini/")

# ─────────────────────────────────────────────────────────
# 2. LIMPIEZA Y ESTANDARIZACIÓN
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 2 — Limpieza y estandarización")
print("=" * 60)

df_clean = df.copy()

# 2.1 Eliminar duplicados
antes = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"► Duplicados eliminados: {antes - len(df_clean)}")

# 2.2 Eliminar outliers extremos usando IQR en variables clave
def eliminar_outliers_iqr(dataframe, columna, factor=3.0):
    Q1 = dataframe[columna].quantile(0.25)
    Q3 = dataframe[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - factor * IQR
    limite_sup = Q3 + factor * IQR
    return dataframe[(dataframe[columna] >= limite_inf) & (dataframe[columna] <= limite_sup)]

cols_outlier = ["habitaciones_prom", "dormitorios_prom", "ocupantes_prom"]
for col in cols_outlier:
    n_antes = len(df_clean)
    df_clean = eliminar_outliers_iqr(df_clean, col)
    print(f"► Outliers en '{col}': {n_antes - len(df_clean)} registros removidos")

# 2.3 Ingeniería de características
df_clean["ratio_dorm_hab"] = df_clean["dormitorios_prom"] / df_clean["habitaciones_prom"]
df_clean["ingreso_edad"]   = df_clean["ingreso_mediano"] * df_clean["edad_casa"]

print(f"\n► Dimensiones tras limpieza: {df_clean.shape}")
print(f"► Tasa de retención: {len(df_clean)/len(df)*100:.1f}%")

# Guardar datos limpios
df_clean.to_csv("datos/datos_limp/california_housing_clean.csv", index=False)
print("✓ Datos limpios guardados en datos/datos_limp/")

# ─────────────────────────────────────────────────────────
# 3. VISUALIZACIONES EXPLORATORIAS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 3 — Visualizaciones exploratorias")
print("=" * 60)

os.makedirs("fuentes/figuras", exist_ok=True)

# 3.1 Distribución de la variable objetivo
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_clean["precio_mediano"], bins=50, color="#2E75B6", edgecolor="white", alpha=0.85)
axes[0].set_title("Distribución: Precio Mediano", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Precio (×$100,000 USD)")
axes[0].set_ylabel("Frecuencia")
axes[1].hist(np.log1p(df_clean["precio_mediano"]), bins=50, color="#1F7D4E", edgecolor="white", alpha=0.85)
axes[1].set_title("Distribución: log(Precio + 1)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(Precio + 1)")
plt.tight_layout()
plt.savefig("fuentes/figuras/distribucion_precio.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 1: distribución del precio guardada")

# 3.2 Correlación
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
ax.set_title("Matriz de Correlación — California Housing", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fuentes/figuras/correlacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 2: matriz de correlación guardada")

# 3.3 Scatter ingreso vs precio
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df_clean["ingreso_mediano"], df_clean["precio_mediano"],
                     alpha=0.3, s=5, c=df_clean["edad_casa"], cmap="viridis")
plt.colorbar(scatter, ax=ax, label="Edad de la casa (años)")
ax.set_xlabel("Ingreso Mediano (×$10,000 USD)")
ax.set_ylabel("Precio Mediano (×$100,000 USD)")
ax.set_title("Ingreso Mediano vs. Precio (color = edad)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fuentes/figuras/ingreso_vs_precio.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 3: scatter ingreso vs precio guardada")

# ─────────────────────────────────────────────────────────
# 4. PREPARACIÓN PARA MODELADO
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 4 — Preparación de features y split")
print("=" * 60)

FEATURES = ["ingreso_mediano", "edad_casa", "habitaciones_prom",
            "dormitorios_prom", "poblacion", "ocupantes_prom",
            "latitud", "longitud", "ratio_dorm_hab", "ingreso_edad"]
TARGET = "precio_mediano"

X = df_clean[FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado (necesario para Ridge)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"► Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
print(f"► Features: {len(FEATURES)}")

# ─────────────────────────────────────────────────────────
# 5. DEFINICIÓN DE FUNCIÓN DE MÉTRICAS
# ─────────────────────────────────────────────────────────
def calcular_metricas(y_true, y_pred, nombre_modelo=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  ── {nombre_modelo} ──")
    print(f"     MAE  = {mae:.4f}")
    print(f"     RMSE = {rmse:.4f}")
    print(f"     R²   = {r2:.4f}")
    print(f"     MAPE = {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# ─────────────────────────────────────────────────────────
# 6. MODELO BASELINE — RIDGE REGRESSION (CV + Grid Search)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 6 — Baseline: Ridge Regression con Grid Search")
print("=" * 60)

param_grid_ridge = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

gs_ridge = GridSearchCV(
    Ridge(), param_grid_ridge,
    cv=kf, scoring="r2", n_jobs=-1, verbose=0
)
gs_ridge.fit(X_train_sc, y_train)

best_ridge      = gs_ridge.best_estimator_
best_alpha      = gs_ridge.best_params_["alpha"]
cv_r2_ridge     = gs_ridge.best_score_

y_pred_ridge = best_ridge.predict(X_test_sc)
metricas_ridge = calcular_metricas(y_test, y_pred_ridge, "Ridge Regression")

print(f"\n► Mejor alpha: {best_alpha}")
print(f"► CV R² (5-fold): {cv_r2_ridge:.4f}")

# ─────────────────────────────────────────────────────────
# 7. MODELO PRINCIPAL — XGBOOST (CV + Grid Search)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 7 — Modelo principal: XGBoost con Grid Search")
print("=" * 60)

param_grid_xgb = {
    "n_estimators":     [100, 200],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
gs_xgb = GridSearchCV(
    xgb_base, param_grid_xgb,
    cv=kf, scoring="r2", n_jobs=-1, verbose=0
)
gs_xgb.fit(X_train, y_train)

best_xgb    = gs_xgb.best_estimator_
best_params = gs_xgb.best_params_
cv_r2_xgb   = gs_xgb.best_score_

y_pred_xgb = best_xgb.predict(X_test)
metricas_xgb = calcular_metricas(y_test, y_pred_xgb, "XGBoost")

print(f"\n► Mejores hiperparámetros XGBoost: {best_params}")
print(f"► CV R² (5-fold): {cv_r2_xgb:.4f}")

# ─────────────────────────────────────────────────────────
# 8. VALIDACIÓN CRUZADA EXTENDIDA (comparación estadística)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 8 — Validación cruzada extendida y prueba t")
print("=" * 60)

cv_scores_ridge = cross_val_score(best_ridge, X_train_sc, y_train, cv=kf, scoring="r2")
cv_scores_xgb   = cross_val_score(best_xgb,   X_train,    y_train, cv=kf, scoring="r2")

print(f"\n► Ridge  CV R²: {cv_scores_ridge.mean():.4f} ± {cv_scores_ridge.std():.4f}")
print(f"► XGBoost CV R²: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

t_stat, p_val = stats.ttest_rel(cv_scores_xgb, cv_scores_ridge)
print(f"\n► Prueba t pareada: t = {t_stat:.4f}, p-value = {p_val:.6f}")
if p_val < 0.05:
    print("  → Diferencia estadísticamente significativa (p < 0.05)")
else:
    print("  → Sin diferencia estadísticamente significativa")

# ─────────────────────────────────────────────────────────
# 9. FIGURA: Comparativa predicciones
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample_idx = np.random.choice(len(y_test), 300, replace=False)
y_sample   = np.array(y_test)[sample_idx]

for ax, preds, titulo, color in zip(
    axes,
    [np.array(y_pred_ridge)[sample_idx], np.array(y_pred_xgb)[sample_idx]],
    ["Ridge Regression", "XGBoost"],
    ["#2E75B6", "#1F7D4E"]
):
    ax.scatter(y_sample, preds, alpha=0.4, s=12, color=color)
    lims = [min(y_sample.min(), preds.min()), max(y_sample.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Predicho")
    ax.set_title(f"{titulo}\nR² = {r2_score(y_sample, preds):.4f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plt.suptitle("Comparativa: Valor Real vs. Predicho", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("fuentes/figuras/comparativa_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Figura 4: comparativa de modelos guardada")

# Figura: Feature importance XGBoost
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(FEATURES)), importances[indices], color="#1F7D4E", edgecolor="white", alpha=0.85)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels([FEATURES[i] for i in indices], rotation=40, ha="right", fontsize=9)
ax.set_title("Importancia de Variables — XGBoost", fontsize=13, fontweight="bold")
ax.set_ylabel("Importancia")
plt.tight_layout()
plt.savefig("fuentes/figuras/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 5: feature importance guardada")

# ─────────────────────────────────────────────────────────
# 10. REGISTRO EN MLFLOW
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" PASO 10 — Registro en MLflow")
print("=" * 60)

# ── Experimento 1: Ridge Regression ──
with mlflow.start_run(run_name="Ridge_Regression_GridSearchCV"):
    mlflow.log_param("modelo",        "Ridge Regression")
    mlflow.log_param("alpha",         best_alpha)
    mlflow.log_param("cv_folds",      5)
    mlflow.log_param("n_features",    len(FEATURES))
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples",  X_test.shape[0])

    mlflow.log_metric("MAE",     metricas_ridge["MAE"])
    mlflow.log_metric("RMSE",    metricas_ridge["RMSE"])
    mlflow.log_metric("R2",      metricas_ridge["R2"])
    mlflow.log_metric("MAPE",    metricas_ridge["MAPE"])
    mlflow.log_metric("CV_R2_mean", cv_scores_ridge.mean())
    mlflow.log_metric("CV_R2_std",  cv_scores_ridge.std())

    mlflow.sklearn.log_model(best_ridge, "ridge_model")
    mlflow.log_artifact("fuentes/figuras/distribucion_precio.png")
    mlflow.log_artifact("fuentes/figuras/correlacion.png")
    mlflow.log_artifact("fuentes/figuras/comparativa_modelos.png")
    print("✓ Ridge registrado en MLflow")

# ── Experimento 2: XGBoost ──
with mlflow.start_run(run_name="XGBoost_GridSearchCV"):
    mlflow.log_param("modelo",             "XGBoost")
    mlflow.log_param("n_estimators",       best_params["n_estimators"])
    mlflow.log_param("max_depth",          best_params["max_depth"])
    mlflow.log_param("learning_rate",      best_params["learning_rate"])
    mlflow.log_param("subsample",          best_params["subsample"])
    mlflow.log_param("colsample_bytree",   best_params["colsample_bytree"])
    mlflow.log_param("cv_folds",           5)
    mlflow.log_param("n_features",         len(FEATURES))
    mlflow.log_param("train_samples",      X_train.shape[0])
    mlflow.log_param("test_samples",       X_test.shape[0])

    mlflow.log_metric("MAE",     metricas_xgb["MAE"])
    mlflow.log_metric("RMSE",    metricas_xgb["RMSE"])
    mlflow.log_metric("R2",      metricas_xgb["R2"])
    mlflow.log_metric("MAPE",    metricas_xgb["MAPE"])
    mlflow.log_metric("CV_R2_mean", cv_scores_xgb.mean())
    mlflow.log_metric("CV_R2_std",  cv_scores_xgb.std())
    mlflow.log_metric("t_stat",  t_stat)
    mlflow.log_metric("p_value", p_val)

    mlflow.xgboost.log_model(best_xgb, "xgboost_model")
    mlflow.log_artifact("fuentes/figuras/feature_importance.png")
    mlflow.log_artifact("fuentes/figuras/comparativa_modelos.png")
    print("✓ XGBoost registrado en MLflow")

# Guardar modelos
joblib.dump(best_ridge, "fuentes/ridge_model.pkl")
joblib.dump(scaler,     "fuentes/scaler.pkl")
best_xgb.save_model("fuentes/xgboost_model.json")

# ─────────────────────────────────────────────────────────
# 11. RESUMEN FINAL
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" RESUMEN COMPARATIVO FINAL")
print("=" * 60)
print(f"{'Métrica':<12} {'Ridge':>12} {'XGBoost':>12} {'Ganador':>10}")
print("-" * 50)
for m in ["MAE", "RMSE", "R2", "MAPE"]:
    v_r = metricas_ridge[m]
    v_x = metricas_xgb[m]
    if m in ["MAE", "RMSE", "MAPE"]:
        ganador = "XGBoost" if v_x < v_r else "Ridge"
    else:
        ganador = "XGBoost" if v_x > v_r else "Ridge"
    print(f"{m:<12} {v_r:>12.4f} {v_x:>12.4f} {ganador:>10}")

print("\n► Deuda técnica identificada:")
print("  - Explorar SMOTE para regresión (outliers residuales)")
print("  - Implementar monitoreo de data drift con Evidently AI")
print("  - Evaluar LightGBM y CatBoost como alternativas")
print("  - Pipeline de reentrenamiento mensual con datos actualizados")
print("\n✅ Script completado exitosamente. Ver panel en: mlflow ui")
