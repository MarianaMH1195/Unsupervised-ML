# Clasificación de Hongos (Mushroom Classification)

Este proyecto consiste en un análisis exploratorio y preprocesamiento de un dataset de hongos para facilitar su clasificación automatizada mediante técnicas de Machine Learning.

## Estructura del Proyecto

*   `data/`: Contiene el dataset original (`agaricus-lepiota.data`) y su descripción (`agaricus-lepiota.names`).
*   `docs/`: Documentación adicional del proyecto.
*   `notebooks/`: Cuadernos de Jupyter con el análisis de datos.
    *   `mushroom.ipynb`: Análisis principal que incluye EDA, limpieza y preprocesamiento.

## Análisis Realizado

El notebook `mushroom.ipynb` realiza las siguientes etapas:

1.  **Carga de Datos**: Importación del dataset con sus respectivos nombres de columnas.
2.  **Limpieza de Datos**:
    *   Identificación y manejo de valores nulos.
    *   Eliminación de la columna `veil-type` por ser constante y no aportar información.
3.  **Preprocesamiento**:
    *   Manejo de valores desconocidos en la columna `stalk-root`.
    *   **One-Hot Encoding**: Aplicado a las características (X) para convertirlas en formato numérico sin jerarquías.
    *   **Label Encoding**: Aplicado a la variable objetivo (clase) para diferenciar entre comestibles y venenosos.
4.  **Análisis Exploratorio de Datos (EDA)**:
    *   Visualización de la distribución de clases.
    *   Análisis de la relación entre el olor (`odor`) y la toxicidad.
    *   **Matriz de Correlación de Cramér's V**: Análisis avanzado de asociaciones entre variables categóricas.

## Requisitos

Para ejecutar el notebook, se requieren las siguientes bibliotecas de Python:

*   streamlit
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   plotly
*   scikit-learn
*   scipy

## Aplicación Streamlit

Este proyecto incluye una aplicación interactiva que permite predecir la comestibilidad de un hongo.

### Cómo ejecutar la aplicación

1.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ejecuta la aplicación desde la raíz del proyecto:
    ```bash
    streamlit run app.py
    ```

## Uso del Notebook

1.  Asegúrate de tener instaladas las dependencias.
2.  Abre el notebook `notebooks/mushroom.ipynb` en Jupyter o Google Colab.
3.  Ejecuta las celdas para reproducir el análisis.
