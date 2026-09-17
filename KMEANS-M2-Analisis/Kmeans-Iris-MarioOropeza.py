"""Análisis del desempeño de K-Means sobre el conjunto Iris.

Mario Oropeza Pérez - A01660605
TC3009C.601
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split


K = 3
RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "resultados"

BASE_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

BASE_PARAMS = {
    "n_clusters": K,
    "init": "k-means++",
    "n_init": 20,
    "max_iter": 300,
    "tol": 0.0001,
    "random_state": RANDOM_STATE,
    "algorithm": "lloyd",
}


FEATURE_SETS = {
    "Todas las variables": BASE_FEATURES,
    "Solo pétalo": ["petal length (cm)", "petal width (cm)"],
    "Solo sépalo": ["sepal length (cm)", "sepal width (cm)"],
    "Longitud y pétalo": [
        "sepal length (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
}


PARAM_OPTIONS = [
    {"init": "k-means++", "n_init": 10, "tol": 0.001},
    {"init": "k-means++", "n_init": 20, "tol": 0.001},
    {"init": "k-means++", "n_init": 50, "tol": 0.001},
    {"init": "k-means++", "n_init": 10, "tol": 0.0001},
    {"init": "k-means++", "n_init": 20, "tol": 0.0001},
    {"init": "k-means++", "n_init": 50, "tol": 0.0001},
    {"init": "random", "n_init": 10, "tol": 0.001},
    {"init": "random", "n_init": 20, "tol": 0.001},
    {"init": "random", "n_init": 50, "tol": 0.001},
    {"init": "random", "n_init": 10, "tol": 0.0001},
    {"init": "random", "n_init": 20, "tol": 0.0001},
    {"init": "random", "n_init": 50, "tol": 0.0001},
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    iris = load_iris(as_frame=True)
    return iris.data.copy(), iris.target.copy(), iris.target_names.tolist()


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def fit_model(x_train: pd.DataFrame, params: dict) -> KMeans:
    model = KMeans(**params)
    model.fit(x_train)
    return model


def build_cluster_mapping(clusters: np.ndarray, y_train: pd.Series) -> dict[int, int]:
    """Relaciona cada cluster con la especie mayoritaria del entrenamiento."""
    y_values = y_train.to_numpy()
    mapping: dict[int, int] = {}

    for cluster_id in range(K):
        labels = y_values[clusters == cluster_id]
        mapping[cluster_id] = int(np.bincount(labels, minlength=K).argmax())

    return mapping


def map_clusters_to_classes(clusters: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(cluster)] for cluster in clusters], dtype=int)


def evaluate_split(
    model: KMeans,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    mapping: dict[int, int],
    train: bool = False,
) -> dict[str, float]:
    clusters = model.labels_ if train else model.predict(x_data)
    y_pred = map_clusters_to_classes(clusters, mapping)

    return {
        "accuracy": accuracy_score(y_data, y_pred),
        "precision_macro": precision_score(y_data, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_data, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_data, y_pred, average="macro", zero_division=0),
        "silhouette": silhouette_score(x_data, clusters),
    }


def evaluate_model(
    model: KMeans,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[int, int]]:
    mapping = build_cluster_mapping(model.labels_, y_train)

    train_metrics = evaluate_split(model, x_train, y_train, mapping, train=True)
    validation_metrics = evaluate_split(model, x_validation, y_validation, mapping)
    test_metrics = evaluate_split(model, x_test, y_test, mapping)

    metrics = pd.DataFrame([
        {"split": "Train", **train_metrics},
        {"split": "Validation", **validation_metrics},
        {"split": "Test", **test_metrics},
    ])

    return metrics, mapping


def search_best_model(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, list[str], dict]:
    rows = []

    for feature_name, features in FEATURE_SETS.items():
        for option in PARAM_OPTIONS:
            params = {
                "n_clusters": K,
                "init": option["init"],
                "n_init": option["n_init"],
                "max_iter": 300,
                "tol": option["tol"],
                "random_state": RANDOM_STATE,
                "algorithm": "lloyd",
            }

            model = fit_model(x_train[features], params)
            mapping = build_cluster_mapping(model.labels_, y_train)
            validation_metrics = evaluate_split(
                model,
                x_validation[features],
                y_validation,
                mapping,
            )

            rows.append({
                "variables": feature_name,
                "init": option["init"],
                "n_init": option["n_init"],
                "tol": option["tol"],
                "accuracy_validation": validation_metrics["accuracy"],
                "f1_validation": validation_metrics["f1_macro"],
                "silhouette_validation": validation_metrics["silhouette"],
            })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["accuracy_validation", "f1_validation", "silhouette_validation", "n_init"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best = results.iloc[0]
    best_features = FEATURE_SETS[best["variables"]]
    best_params = {
        "n_clusters": K,
        "init": best["init"],
        "n_init": int(best["n_init"]),
        "max_iter": 300,
        "tol": float(best["tol"]),
        "random_state": RANDOM_STATE,
        "algorithm": "lloyd",
    }

    return results, best_features, best_params


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=range(K))
    display = ConfusionMatrixDisplay(matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_comparison(base_metrics: pd.DataFrame, adjusted_metrics: pd.DataFrame) -> None:
    labels = ["Train", "Validation", "Test"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, base_metrics["accuracy"], width, label="Modelo base")
    ax.bar(x + width / 2, adjusted_metrics["accuracy"], width, label="Modelo ajustado")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy antes y después del ajuste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparacion_accuracy_base_vs_ajustado.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_comparison(search_results: pd.DataFrame) -> None:
    summary = (
        search_results.groupby("variables", as_index=False)["accuracy_validation"]
        .max()
        .sort_values("accuracy_validation", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(summary["variables"], summary["accuracy_validation"])
    ax.set_ylim(0.60, 1.03)
    ax.set_ylabel("Accuracy en Validation")
    ax.set_title("Comparación de variables en Validation")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparacion_variables_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_curve(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    features: list[str],
    params: dict,
    filename: str,
) -> pd.DataFrame:
    sizes = [30, 45, 60, 75, 90]
    rows = []

    for size in sizes:
        subset = x_train.iloc[:size]
        y_subset = y_train.iloc[:size]
        model = fit_model(subset[features], params)
        mapping = build_cluster_mapping(model.labels_, y_subset)

        train_metrics = evaluate_split(model, subset[features], y_subset, mapping, train=True)
        validation_metrics = evaluate_split(
            model,
            x_validation[features],
            y_validation,
            mapping,
        )

        rows.append({
            "observaciones_train": size,
            "accuracy_train": train_metrics["accuracy"],
            "accuracy_validation": validation_metrics["accuracy"],
        })

    curve = pd.DataFrame(rows)
    curve.to_csv(OUTPUT_DIR / f"{filename}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(curve["observaciones_train"], curve["accuracy_train"], marker="o", label="Train")
    ax.plot(curve["observaciones_train"], curve["accuracy_validation"], marker="o", label="Validation")
    ax.set_ylim(0.60, 1.03)
    ax.set_xlabel("Observaciones de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title("Curva de aprendizaje")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return curve


def get_diagnosis(metrics: pd.DataFrame) -> dict[str, str | float]:
    train_accuracy = float(metrics.loc[metrics["split"] == "Train", "accuracy"].iloc[0])
    validation_accuracy = float(metrics.loc[metrics["split"] == "Validation", "accuracy"].iloc[0])
    test_accuracy = float(metrics.loc[metrics["split"] == "Test", "accuracy"].iloc[0])

    bias_value = 1 - train_accuracy
    max_gap = max(abs(train_accuracy - validation_accuracy), abs(train_accuracy - test_accuracy))

    if train_accuracy >= 0.95:
        bias = "Bajo"
    elif train_accuracy >= 0.85:
        bias = "Medio"
    else:
        bias = "Alto"

    if max_gap <= 0.07:
        variance = "Baja"
    elif max_gap <= 0.15:
        variance = "Media"
    else:
        variance = "Alta"

    if train_accuracy < 0.92 and max_gap <= 0.07:
        fit = "Underfit leve"
    elif train_accuracy >= 0.92 and max_gap <= 0.07:
        fit = "Fit"
    else:
        fit = "Overfit"

    return {
        "bias": bias,
        "bias_proxy": round(bias_value, 4),
        "varianza": variance,
        "brecha_maxima": round(max_gap, 4),
        "ajuste": fit,
    }


def save_predictions(
    model: KMeans,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    mapping: dict[int, int],
    target_names: list[str],
    features: list[str],
) -> np.ndarray:
    clusters = model.predict(x_test[features])
    y_pred = map_clusters_to_classes(clusters, mapping)

    return y_pred


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y, target_names = load_dataset()
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(x, y)

    base_model = fit_model(x_train[BASE_FEATURES], BASE_PARAMS)
    base_metrics, base_mapping = evaluate_model(
        base_model,
        x_train[BASE_FEATURES],
        x_validation[BASE_FEATURES],
        x_test[BASE_FEATURES],
        y_train,
        y_validation,
        y_test,
    )

    search_results, best_features, best_params = search_best_model(
        x_train,
        x_validation,
        y_train,
        y_validation,
    )

    adjusted_model = fit_model(x_train[best_features], best_params)
    adjusted_metrics, adjusted_mapping = evaluate_model(
        adjusted_model,
        x_train[best_features],
        x_validation[best_features],
        x_test[best_features],
        y_train,
        y_validation,
        y_test,
    )

    base_pred = save_predictions(
        base_model,
        x_test,
        y_test,
        base_mapping,
        target_names,
        BASE_FEATURES,
    )
    adjusted_pred = save_predictions(
        adjusted_model,
        x_test,
        y_test,
        adjusted_mapping,
        target_names,
        best_features,
    )

    base_metrics.round(4).to_csv(OUTPUT_DIR / "metricas_modelo_base.csv", index=False)
    adjusted_metrics.round(4).to_csv(OUTPUT_DIR / "metricas_modelo_ajustado.csv", index=False)
    search_results.round(4).to_csv(OUTPUT_DIR / "busqueda_validation.csv", index=False)

    predictions = pd.DataFrame({
        "indice_original": x_test.index,
        "especie_real": [target_names[int(v)] for v in y_test],
        "prediccion_base": [target_names[int(v)] for v in base_pred],
        "prediccion_ajustada": [target_names[int(v)] for v in adjusted_pred],
    })
    predictions.to_csv(OUTPUT_DIR / "predicciones_test.csv", index=False)

    save_confusion_matrix(
        y_test,
        base_pred,
        target_names,
        "Matriz de confusión - modelo base",
        OUTPUT_DIR / "matriz_confusion_base.png",
    )
    save_confusion_matrix(
        y_test,
        adjusted_pred,
        target_names,
        "Matriz de confusión - modelo ajustado",
        OUTPUT_DIR / "matriz_confusion_ajustado.png",
    )
    save_accuracy_comparison(base_metrics, adjusted_metrics)
    save_feature_comparison(search_results)

    learning_curve(
        x_train,
        y_train,
        x_validation,
        y_validation,
        BASE_FEATURES,
        BASE_PARAMS,
        "curva_aprendizaje_base",
    )
    learning_curve(
        x_train,
        y_train,
        x_validation,
        y_validation,
        best_features,
        best_params,
        "curva_aprendizaje_ajustado",
    )

    summary = {
        "dataset": "Iris",
        "train": len(x_train),
        "validation": len(x_validation),
        "test": len(x_test),
        "modelo_base": {
            "variables": BASE_FEATURES,
            "parametros": BASE_PARAMS,
            "diagnostico": get_diagnosis(base_metrics),
        },
        "modelo_ajustado": {
            "variables": best_features,
            "parametros": best_params,
            "diagnostico": get_diagnosis(adjusted_metrics),
        },
    }

    with open(OUTPUT_DIR / "resumen_analisis.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("K-Means sobre Iris - análisis de desempeño")
    print(f"Train: {len(x_train)}")
    print(f"Validation: {len(x_validation)}")
    print(f"Test: {len(x_test)}")
    print("\nModelo base")
    print(base_metrics.round(4).to_string(index=False))
    print("\nModelo ajustado")
    print(adjusted_metrics.round(4).to_string(index=False))
    print(f"\nVariables seleccionadas: {', '.join(best_features)}")
    print(f"Parámetros seleccionados: {best_params}")
    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
