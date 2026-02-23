# 🍄 Clasificación de Setas (Mushroom Dataset) - Análisis No Supervisado y Supervisado

Este repositorio contiene un proyecto práctico de análisis de datos y modelado utilizando técnicas de **Aprendizaje Automático No Supervisado** (PCA y K-Means Clustering) y **Supervisado** (Random Forest), centrado en el estudio del "Mushroom Dataset".

## 📂 Dataset

El conjunto de datos utilizado es el **Mushroom Dataset** del repositorio UCI. Representa muestras de hongos correspondientes a 23 especies de setas de láminas.

*   **Variables**: 22 características categóricas (forma del sombrero, color, olor, etc.).
*   **Variable Objetivo (`class`)**: Binaria — `e` (comestible) o `p` (venenoso).
*   **Origen**: [Mushroom Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)

## 🧠 Objetivos del Proyecto

*   Explorar y limpiar un dataset categórico complejo.
*   Implementar preprocesamiento avanzado: tratamiento de valores nulos (` stalk-root`) y eliminación de columnas constantes.
*   Codificación de variables mediante **One-Hot Encoding** y **Label Encoding**.
*   Reducción de dimensionalidad utilizando **Análisis de Componentes Principales (PCA)**.
*   Segmentación de datos mediante **Clustering (K-Means)** para detectar patrones ocultos.
*   Comparativa de rendimiento con un modelo supervisado de **Random Forest**.

## 🔧 Tecnologías Utilizadas

*   **Lenguaje**: Python
*   **Análisis de Datos**: Pandas, NumPy
*   **Visualización**: Seaborn, Matplotlib, Plotly
*   **Machine Learning**: Scikit-learn (PCA, KMeans, RandomForestClassifier)
*   **Estadística**: Scipy (Cramér's V para análisis de correlación categórica)

## 🗂️ Estructura del Proyecto

*   `data/`: Archivos originales del dataset.
*   `docs/`: Documentación adicional y diccionario de datos.
*   `notebooks/`: Jupyter Notebook `mushroom.ipynb` con todo el código y análisis.

## 🧪 Análisis Destacados en el Notebook

1.  **Análisis de Relación Categórica**: Implementación de la matriz de correlación basada en el **V de Cramér**.
2.  **PCA**: Reducción a 2 componentes para visualizar la separabilidad de las clases en un plano 2D.
3.  **Evaluación de Modelos**: Comparación entre agrupamiento natural (Clustering) y clasificación dirigida (Random Forest), analizando precisión y méticas de error.

## 📊 Evaluación de Competencias

✅ Uso y gestión de formato .csv  
✅ Limpieza y preprocesado (Imputación de faltantes y codificación)  
✅ Visualización avanzada (Heatmaps, PCA Scatter plots)  
✅ Reducción de dimensionalidad (PCA)  
✅ Modelado No Supervisado (K-Means)  
✅ Modelado Supervisado (Ensemble: Random Forest)  
✅ Análisis exploratorio detallado (EDA)  

---
**Nota**: Para ejecutar el notebook en entornos locales, asegúrese de tener configuradas las rutas relativas correctamente (`../data/agaricus-lepiota.data`).
