"""
datos_prep.py — Funciones de limpieza y preparación de datos
Proyecto: Predicción de Precios de Vivienda | Actividad 5
Alumno: Kevin Espindola Bravo | AL02963812
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


def cargar_datos_brutos() -> pd.DataFrame:
    """Carga el dataset California Housing y renombra columnas al español."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    col_rename = {
        "MedInc":      "ingreso_mediano",
        "HouseAge":    "edad_casa",
        "AveRooms":    "habitaciones_prom",
        "AveBedrms":   "dormitorios_prom",
        "Population":  "poblacion",
        "AveOccup":    "ocupantes_prom",
        "Latitude":    "latitud",
        "Longitude":   "longitud",
        "MedHouseVal": "precio_mediano",
    }
    df.rename(columns=col_rename, inplace=True)
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros exactamente duplicados."""
    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[limpieza] Duplicados eliminados: {n_antes - len(df)}")
    return df


def eliminar_outliers_iqr(df: pd.DataFrame, columna: str, factor: float = 3.0) -> pd.DataFrame:
    """Elimina outliers extremos de una columna usando el método IQR."""
    Q1, Q3 = df[columna].quantile(0.25), df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - factor * IQR, Q3 + factor * IQR
    n_antes = len(df)
    df = df[(df[columna] >= lim_inf) & (df[columna] <= lim_sup)].reset_index(drop=True)
    print(f"[limpieza] Outliers en '{columna}': {n_antes - len(df)} registros eliminados")
    return df


def ingenieria_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea nuevas variables derivadas relevantes para el modelo."""
    df = df.copy()
    df["ratio_dorm_hab"] = df["dormitorios_prom"] / df["habitaciones_prom"]
    df["ingreso_edad"]   = df["ingreso_mediano"] * df["edad_casa"]
    print("[features] Variables nuevas: ratio_dorm_hab, ingreso_edad")
    return df


def pipeline_limpieza(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el pipeline completo de limpieza sobre el dataframe."""
    df = eliminar_duplicados(df)
    for col in ["habitaciones_prom", "dormitorios_prom", "ocupantes_prom"]:
        df = eliminar_outliers_iqr(df, col)
    df = ingenieria_caracteristicas(df)
    return df


def reporte_calidad(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Imprime un reporte de calidad comparando raw vs clean."""
    print("\n─── Reporte de Calidad de Datos ───")
    print(f"  Registros originales : {len(df_raw):,}")
    print(f"  Registros limpios    : {len(df_clean):,}")
    print(f"  Tasa de retención    : {len(df_clean)/len(df_raw)*100:.1f}%")
    print(f"  Valores nulos (clean): {df_clean.isnull().sum().sum()}")
    print(f"  Variables totales    : {df_clean.shape[1]}")
    print("───────────────────────────────────\n")


if __name__ == "__main__":
    import os
    df_raw   = cargar_datos_brutos()
    df_clean = pipeline_limpieza(df_raw)
    reporte_calidad(df_raw, df_clean)

    os.makedirs("datos/datos_ini",  exist_ok=True)
    os.makedirs("datos/datos_limp", exist_ok=True)
    df_raw.to_csv("datos/datos_ini/california_housing_raw.csv",   index=False)
    df_clean.to_csv("datos/datos_limp/california_housing_clean.csv", index=False)
    print("✓ Archivos CSV guardados.")
