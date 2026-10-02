"""
Clasificador de textos por ODS (Objetivos de Desarrollo Sostenible)
====================================================================

Esta aplicación NO entrena ningún modelo: carga el pipeline ya entrenado
y exportado con `joblib` desde `MicroProyecto2G30.ipynb`
(`models/pipeline_ods.joblib`) y lo usa directamente para predecir:

    - Entrada de texto
    - TF-IDF (bolsa de palabras) con `ods_preprocessing.text_preprocess`
    - LSA (TruncatedSVD)
    - Regresión logística (hiperparámetros elegidos con GridSearchCV)

Dado que los datos de entrenamiento únicamente contemplan las iniciativas
1 a la 16, el modelo no puede predecir la clase 17 (Alianzas para lograr
los objetivos).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# El pipeline serializado referencia `ods_preprocessing.text_preprocess`,
# por lo que el módulo debe poder importarse antes de cargarlo.
import ods_preprocessing  # noqa: F401

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pipeline_ods.joblib"

ODS_NAMES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

st.set_page_config(page_title="Clasificador ODS", page_icon="🌍", layout="centered")


@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline_clf = load_pipeline()

st.title("🌍 Clasificador de textos por ODS")
st.write(
    "Ingresa un texto libre en español y el modelo predecirá con cuál de los "
    "**Objetivos de Desarrollo Sostenible (ODS)** se relaciona"
)

user_text = st.text_area(
    "Texto a clasificar",
    height=180,
)

clasificar = st.button("Clasificar", type="primary")

if clasificar:
    clean_text = (user_text or "").strip()
    if not clean_text:
        st.warning("Por favor ingresa un texto antes de clasificar.")
    else:
        proba = pipeline_clf.predict_proba([clean_text])[0]
        classes = pipeline_clf.classes_
        order = np.argsort(proba)[::-1]

        ods_pred = int(classes[order[0]])
        ods_trust = proba[order[0]]

        st.subheader("Resultado")
        st.success(
            f"**ODS {ods_pred} — {ODS_NAMES[ods_pred]}** "
            f"(confianza: {ods_trust:.1%})"
        )

        st.write("Probabilidades por ODS (top 5):")
        top5_idx = order[:5]
        resultados_df = pd.DataFrame({
            "ODS": [f"{int(classes[i])} - {ODS_NAMES[int(classes[i])]}" for i in top5_idx],
            "Probabilidad": [proba[i] for i in top5_idx],
        }).set_index("ODS")
        st.bar_chart(resultados_df)
