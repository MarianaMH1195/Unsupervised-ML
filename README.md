# 🍄 Mushroom Intel Dashboard

Este proyecto consiste en un análisis avanzado y una aplicación interactiva (Dashboard) diseñada para clasificar especies de hongos como **comestibles** o **venenosos** basándose en sus características físicas. Combina técnicas de Machine Learning con una interfaz de usuario profesional y moderna.

## 🚀 Aplicación Streamlit (v2)

La joya de la corona de este proyecto es el **Mushroom Intel Dashboard**, una interfaz de alta gama que ofrece:

*   **📊 Dashboard Principal**: Resumen visual con métricas clave y proporciones globales de especies mediante gráficos interactivos.
*   **🔍 Análisis EDA Dinámico**: Herramienta de exploración que permite visualizar la distribución de cualquier característica del hongo en tiempo real.
*   **🧠 Predictor con IA**: Un formulario optimizado que utiliza un modelo **RandomForestClassifier** para determinar la toxicidad de un hongo con un alto nivel de confianza.
*   **🎨 Interfaz Télica**: Diseño personalizado con una paleta de colores moderno (Verde Teal y Naranja), navegación lateral profesional y experiencia de usuario fluida.

### Cómo ejecutar la aplicación

1.  **Instala las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ejecuta el Dashboard**:
    ```bash
    streamlit run app.py
    ```

---

## 📂 Estructura del Proyecto

*   `data/`: Contiene el dataset original (`agaricus-lepiota.data`).
*   `docs/`: Documentación técnica del proyecto.
*   `notebooks/`: Cuadernos de Jupyter con el análisis de datos original.
    *   `mushroom.ipynb`: EDA, limpieza, preprocesamiento y matriz de Cramér's V.
*   `app.py`: Código fuente de la aplicación principal en Streamlit.
*   `requirements.txt`: Lista de dependencias del proyecto.

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
