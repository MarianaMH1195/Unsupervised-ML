import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Mushroom Edibility Predictor", page_icon="🍄", layout="wide")

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍄 Clasificador de Hongos: ¿Comestible o Venenoso?")
st.markdown("Esta aplicación predice si un hongo es **comestible** o **venenoso** basándose en sus características físicas.")

@st.cache_data
def load_data():
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    df = pd.read_csv("data/agaricus-lepiota.data", names=columns)
    
    # Preprocesamiento según el notebook
    df['stalk-root'] = df['stalk-root'].replace('?', 'unknown')
    df.drop('veil-type', axis=1, inplace=True)
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("No se encontró el archivo de datos en 'data/agaricus-lepiota.data'. Por favor, asegúrese de que el archivo existe.")
    st.stop()

# Entrenamiento del modelo
@st.cache_resource
def train_model(data):
    X = data.drop('class', axis=1)
    y = data['class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_encoded = pd.get_dummies(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y_encoded)
    
    return model, X_encoded.columns, le

model, model_columns, label_encoder = train_model(df)

# Sidebar para entradas del usuario
st.sidebar.header("Características del Hongo")

def user_input_features():
    inputs = {}
    # Obtenemos los valores únicos para cada columna (excepto 'class')
    for col in df.columns:
        if col == 'class':
            continue
        unique_vals = sorted(df[col].unique())
        inputs[col] = st.sidebar.selectbox(f"{col}", unique_vals)
    
    features = pd.DataFrame(inputs, index=[0])
    return features

input_df = user_input_features()

# Mostrar datos de entrada
st.subheader("Características Seleccionadas")
st.write(input_df)

# Predicción
if st.button("Predecir"):
    # Preprocesar la entrada
    input_encoded = pd.get_dummies(input_df)
    
    # Asegurar que tiene todas las columnas del modelo
    full_input = pd.DataFrame(columns=model_columns)
    full_input = pd.concat([full_input, input_encoded], axis=0).fillna(0)
    full_input = full_input[model_columns] # Reordenar columnas
    
    prediction = model.predict(full_input)
    prediction_proba = model.predict_proba(full_input)
    
    res = label_encoder.inverse_transform(prediction)[0]
    
    st.subheader("Resultado de la Predicción")
    if res == 'e':
        st.success("✅ El hongo es **COMESTIBLE**")
    else:
        st.error("⚠️ El hongo es **VENENOSO**")
    
    st.write(f"Confianza: {np.max(prediction_proba)*100:.2f}%")

# Visualizaciones (opcional)
if st.checkbox("Mostrar Análisis de Datos"):
    st.subheader("Distribución de Clases")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='class', palette='viridis', ax=ax)
    ax.set_xticklabels(['Venenoso (p)', 'Comestible (e)'])
    st.pyplot(fig)
    
    st.subheader("Relación Olor vs Clase")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='odor', hue='class', ax=ax2)
    st.pyplot(fig2)
