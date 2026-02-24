import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Mushroom Intel Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO ---
st.markdown("""
    <style>
    /* Fondo principal modo oscuro/profesional */
    .stApp {
        background-color: #0E1117;
    }
    /* Estilo de la navegación lateral */
    section[data-testid="stSidebar"] {
        background-color: #1A1C24;
    }
    /* Botones y acentos */
    .stButton>button {
        background-color: #1ABC9C;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #16A085;
        border: none;
        color: #f0f0f0;
    }
    /* Contenedores de métricas */
    [data-testid="stMetricValue"] {
        color: #1ABC9C;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA Y PROCESAMIENTO DE DATOS ---
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
    df['stalk-root'] = df['stalk-root'].replace('?', np.nan)
    df_clean = df.drop('veil-type', axis=1) # Siempre es constante
    return df_clean, columns

df, original_columns = load_data()

# --- MODELO ---
@st.cache_resource
def train_model(data):
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Encoding de la variable objetivo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Preprocesamiento de características (Ordinal -> KNNImputer)
    # Guardamos los encoders por columna para poder transformar nuevos datos
    encoders = {}
    X_encoded = X.copy()
    for col in X.columns:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Ajustamos solo con los valores que NO son NaN
        non_nan_mask = X[col].notna()
        if non_nan_mask.any():
            oe.fit(X.loc[non_nan_mask, [col]])
            X_encoded.loc[non_nan_mask, col] = oe.transform(X.loc[non_nan_mask, [col]]).flatten()
        X_encoded[col] = pd.to_numeric(X_encoded[col])
        encoders[col] = oe
        
    # Imputación KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_encoded)
    X_final = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Entrenamiento
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_final, y_encoded)
    
    return model, X.columns, le, encoders, imputer

model, features, label_encoder, feature_encoders, knn_imputer = train_model(df)

# --- NAVEGACIÓN LATERAL ---
st.sidebar.image("docs/mushroom_logo.png", use_container_width=True)
st.sidebar.markdown("---") # <--- recuadro de nota en la barra lateral de IA
menu = st.sidebar.radio(
    "Menú Principal",
    ["Dashboard Principal", "Análisis Exploratorio (EDA)", "Predictor de Especies"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("Este dashboard utiliza IA para analizar y clasificar especies de hongos basadas en el dataset de UCI Machine Learning.")

# --- SECCIONES ---
if menu == "Dashboard Principal":
    st.image("docs/mushroom_banner.png", use_container_width=True)
    st.title("Dashboard de Visión General")
    st.markdown("---")
    
    # Métricas clave
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Especies", len(df))
    m2.metric("Comestibles", len(df[df['class']=='e']), delta_color="normal")
    m3.metric("Venenosas", len(df[df['class']=='p']), delta_color="inverse")
    m4.metric("Características", len(df.columns)-1)
    
    tab1, tab2 = st.tabs(["🌎 Distribución Global", "📦 Perfil de Características"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Gráfico Circular: Distribución de Clases
            fig_pie = px.pie(df, names='class', title="Proporción Comestibles vs Venenosos",
                            color='class', color_discrete_map={'e':'#1ABC9C', 'p':'#E67E22'},
                            labels={'class': 'Clasificación'})
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gráfico de Barras: Hábitat
            habitat_dist = df.groupby(['habitat', 'class']).size().reset_index(name='counts')
            fig_hab = px.bar(habitat_dist, x='habitat', y='counts', color='class',
                             title="Distribución por Hábitat", barmode='group',
                             color_discrete_map={'e':'#1ABC9C', 'p':'#E67E22'})
            st.plotly_chart(fig_hab, use_container_width=True)

    with tab2:
        # Gráfico por Olor
        odor_dist = df.groupby(['odor', 'class']).size().reset_index(name='counts')
        fig_odor = px.bar(odor_dist, x='odor', y='counts', color='class',
                         title="Relación entre Olor y Clasificación",
                         color_discrete_map={'e':'#1ABC9C', 'p':'#E67E22'})
        st.plotly_chart(fig_odor, use_container_width=True)

elif menu == "Análisis Exploratorio (EDA)":
    st.title("Análisis Detallado de Características")
    st.markdown("---")
    
    feature_to_plot = st.selectbox("Seleccione característica para analizar:", df.columns[1:])
    
    colA, colB = st.columns([1, 2])
    
    with colA:
        st.markdown(f"### Análisis de `{feature_to_plot}`")
        st.write(df.groupby([feature_to_plot, 'class']).size().unstack().fillna(0))
        
    with colB:
        fig_dynamic = px.histogram(df, x=feature_to_plot, color='class', barmode='group',
                                 title=f"Distribución de {feature_to_plot} por Clase",
                                 color_discrete_map={'e':'#1ABC9C', 'p':'#E67E22'})
        st.plotly_chart(fig_dynamic, use_container_width=True)

elif menu == "Predictor de Especies":
    st.title("Inteligencia Artificial: Predictor")
    st.markdown("---")
    st.markdown("Complete las características del hongo encontrado para determinar su seguridad.")
    
    # Formulario de entrada
    with st.expander("Formulario de Características", expanded=True):
        c1, c2, c3 = st.columns(3)
        user_inputs = {}
        
        # Agrupando inputs
        columns_list = list(df.columns)
        columns_list.remove('class')
        
        for i, col in enumerate(columns_list):
            val = sorted(df[col].unique())
            if i % 3 == 0:
                with c1: user_inputs[col] = st.selectbox(f"{col}", val)
            elif i % 3 == 1:
                with c2: user_inputs[col] = st.selectbox(f"{col}", val)
            else:
                with c3: user_inputs[col] = st.selectbox(f"{col}", val)

    if st.button("Ejecutar Predicción con KNN + RandomForest"):
        input_df = pd.DataFrame([user_inputs])
        
        # Procesamiento para el modelo (mismo flujo que el entrenamiento)
        input_encoded = input_df.copy()
        for col in features:
            oe = feature_encoders[col]
            # Si el usuario seleccionó 'nan' (que no debería en el selector actual, pero por consistencia)
            if input_df[col].iloc[0] == 'nan' or pd.isna(input_df[col].iloc[0]):
                input_encoded[col] = np.nan
            else:
                input_encoded[col] = oe.transform(input_df[[col]]).flatten()[0]
        
        # Imputación KNN sobre la entrada (aunque el selector no tenga NaNs ahora, permite escalabilidad)
        input_imputed = knn_imputer.transform(input_encoded)
        
        prediction = model.predict(input_imputed)
        prob = model.predict_proba(input_imputed)
        res_label = label_encoder.inverse_transform(prediction)[0]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if res_label == 'e':
                st.success("### ✅ RESULTADO: COMESTIBLE")
                st.balloons()
            else:
                st.error("### ⚠️ RESULTADO: VENENOSO")
        
        with res_col2:
            st.metric("Confianza del Modelo", f"{np.max(prob)*100:.2f}%")
            
        # Gráfico de confianza
        fig_prob = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = np.max(prob)*100,
            title = {'text': "Nivel de Certeza (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1ABC9C"},
                'steps' : [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}]
            }
        ))
        st.plotly_chart(fig_prob, use_container_width=True)
