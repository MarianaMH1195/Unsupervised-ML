# 🍄 Mushroom Intel Dashboard

Este proyecto consiste en un análisis avanzado y una aplicación interactiva (Dashboard) diseñada para clasificar especies de hongos como **comestibles** o **venenosos** basándose en sus características físicas. Combina técnicas de Machine Learning con una interfaz de usuario profesional y moderna.

## 🚀 Aplicación Streamlit (v2)

La joya de la corona de este proyecto es el **Mushroom Intel Dashboard**, una interfaz de alta gama que ofrece:

*   **📊 Dashboard Principal**: Resumen visual con métricas clave y proporciones globales de especies mediante gráficos interactivos.
*   **🔍 Análisis EDA Dinámico**: Herramienta de exploración que permite visualizar la distribución de cualquier característica del hongo en tiempo real.
*   **🧠 Predictor con IA**: Un formulario optimizado que utiliza un modelo **RandomForestClassifier** para determinar la toxicidad de un hongo con un alto nivel de confianza.
*   **🎨 Interfaz Télica**: Diseño personalizado con una paleta de colores moderno (Verde Teal y Naranja), navegación lateral profesional y experiencia de usuario fluida.

## Estructura del Proyecto
```text
Unsupervised-ML/
├── data/                       # Dataset original y procesado
│   └── agaricus-lepiota.data
├── notebooks/                  # Análisis exploratorio y prototipado
│   └── mushroom.ipynb
├── docs/                       # Documentación adicional e imágenes
├── app.py                      # Aplicación principal (Streamlit)
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Configuración de archivos excluidos
└── README.md                   # Documentación principal
```

## Características de la Aplicación
- **Preprocesamiento Inteligente**: Uso de `KNNImputer` para manejar valores faltantes en la característica `stalk-root`, basándose en los 5 vecinos más cercanos.
- **Modelado Robusto**: Clasificación mediante `RandomForestClassifier` optimizado.
- **Visualización Interactiva**: Gráficos dinámicos con Plotly para análisis de hábitat, color y olor.

## Instalación y Uso

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/Bootcamp-Data-Analyst/Unsupervised-ML.git
    cd Unsupervised-ML
    ```

2.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación**:
    ```bash
    streamlit run app.py
    ```

---
*Desarrollado como parte del proyecto de análisis de aprendizaje no supervisado y clasificación.*

## 🧪 Análisis Realizado

El núcleo analítico se basa en el notebook `mushroom.ipynb`, siguiendo estas etapas:

1.  **Limpieza y Preprocesamiento**: Manejo de valores nulos, eliminación de columnas constantes (`veil-type`) y codificación de variables categóricas (One-Hot & Label Encoding).
2.  **EDA Avanzado**: Análisis de distribuciones y relaciones entre variables como el olor y la clase.
3.  **Matriz de Cramér's V**: Medición de la asociación entre características categóricas.

## 🛠️ Requisitos

Para asegurar el correcto funcionamiento, se requieren:
*   `streamlit`
*   `pandas`
*   `numpy`
*   `plotly`
*   `scikit-learn`
*   `matplotlib` / `seaborn`

---

## 📝 Uso del Notebook

1.  Asegúrate de tener instaladas las dependencias.
2.  Abre `notebooks/mushroom.ipynb` en Jupyter o Google Colab para revisar el análisis estadístico detallado.
