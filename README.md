# K-Means sobre Iris - análisis de desempeño

Análisis del desempeño de la implementación de **K-Means con scikit-learn** sobre el conjunto Iris. Se utiliza una separación Train / Validation / Test para revisar bias, varianza, nivel de ajuste y el efecto de una nueva configuración del modelo.

**Alumno:** Mario Oropeza Pérez  
**Matrícula:** A01660605  
**Curso:** TC3009C.601

## Estructura del proyecto

```text
KMEANS-M2-Analisis/
├── Kmeans-Iris-MarioOropeza.py
├── Reporte_Analisis_KMeans_Mario_Oropeza.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── resultados/
    ├── busqueda_validation.csv
    ├── comparacion_accuracy_base_vs_ajustado.png
    ├── comparacion_variables_validation.png
    ├── curva_aprendizaje_base.csv
    ├── curva_aprendizaje_base.png
    ├── curva_aprendizaje_ajustado.csv
    ├── curva_aprendizaje_ajustado.png
    ├── matriz_confusion_base.png
    ├── matriz_confusion_ajustado.png
    ├── metricas_modelo_base.csv
    ├── metricas_modelo_ajustado.csv
    ├── predicciones_test.csv
    └── resumen_analisis.json
```

## Dataset

Se utiliza el conjunto **Iris** incluido en `sklearn.datasets.load_iris`. Contiene 150 observaciones con cuatro variables numéricas y tres especies con 50 observaciones cada una.

La separación utilizada es:

- Train: 90 observaciones (60%)
- Validation: 30 observaciones (20%)
- Test: 30 observaciones (20%)

La división es estratificada y utiliza `random_state = 42`.

## Modelo base

Se conserva la configuración de la implementación anterior:

```text
k = 3
init = k-means++
n_init = 20
max_iter = 300
tol = 0.0001
random_state = 42
algorithm = lloyd
```

Resultados:

| Conjunto | Accuracy | F1 macro | Silhouette |
|---|---:|---:|---:|
| Train | 88.89% | 88.57% | 0.5755 |
| Validation | 93.33% | 93.27% | 0.5625 |
| Test | 86.67% | 86.53% | 0.4723 |

Diagnóstico: **bias medio, varianza baja y underfit leve**.

## Ajuste

La selección se realiza usando únicamente el conjunto de Validation. Se comparan distintas inicializaciones, valores de `n_init`, tolerancias y subconjuntos de variables.

La configuración seleccionada fue:

```text
variables = petal length, petal width
k = 3
init = k-means++
n_init = 50
max_iter = 300
tol = 0.001
random_state = 42
algorithm = lloyd
```

Resultados:

| Conjunto | Accuracy | F1 macro | Silhouette |
|---|---:|---:|---:|
| Train | 96.67% | 96.67% | 0.6705 |
| Validation | 100.00% | 100.00% | 0.6450 |
| Test | 93.33% | 93.33% | 0.6029 |

Diagnóstico final: **bias bajo, varianza baja y fit**.

## Ejecución

Crear un entorno e instalar las dependencias:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Ejecutar el programa:

```bash
python Kmeans-Iris-MarioOropeza.py
```

El script genera automáticamente las tablas y gráficas dentro de `resultados/`.

El análisis completo se encuentra en `Reporte_Analisis_KMeans_Mario_Oropeza.pdf`.
