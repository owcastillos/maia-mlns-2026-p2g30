"""
Clasificador de textos por ODS (Objetivos de Desarrollo Sostenible)
====================================================================

Esta aplicación implementa exactamente el mismo pipeline construido
y justificado en `MicroProyecto2G30.ipynb`:

    - Entrada de texto
    - TF-IDF (bolsa de palabras)
    - LSA (TruncatedSVD, 200 componentes)
    - Regresión logística (C=10, class_weight=None)

Dado que los datos de entrenamiento únicamente contemplan las iniciativas 
1 a la 16, el modelo no puede predecir la clase 17 (Alianzas para lograr
los objetivos).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
RANDOM_STATE = 32
N_COMPONENTS_CLF = 200
DATA_PATH = Path(__file__).resolve().parent / "data" / "Datos_textosODS.xlsx"

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

st.set_page_config(page_title="Clasificador ODS", page_icon="🚧", layout="centered")

@st.cache_resource(show_spinner=False)
def _ensure_nltk_data():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)
    return True

_ensure_nltk_data()

_tokenizer = RegexpTokenizer(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+")
_spanish_stopwords = set(stopwords.words("spanish"))
_stemmer = SnowballStemmer("spanish")

def text_preprocess(text: str) -> str:
    tokens = _tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t not in _spanish_stopwords and len(t) > 2]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

@st.cache_resource(show_spinner = "Entrenando el modelo...")
def load_pipeline() -> Pipeline:
    data = pd.read_excel(DATA_PATH)
    data = data.dropna(subset=["textos", "ODS"]).drop_duplicates(subset=["textos"]).reset_index(drop=True)

    X = data["textos"]
    y = data["ODS"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    tfidf_vectorizer = TfidfVectorizer(
        preprocessor=text_preprocess,
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
    )

    pipeline_clf = Pipeline([
        ("tfidf", tfidf_vectorizer),
        ("svd", TruncatedSVD(n_components = N_COMPONENTS_CLF, random_state = RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=10, class_weight=None)),
    ])
    pipeline_clf.fit(X_train, y_train)
    return pipeline_clf


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
        pipeline_clf = load_pipeline()

        proba = pipeline_clf.predict_proba([clean_text])[0]
        classes = pipeline_clf.named_steps["clf"].classes_
        order = np.argsort(proba)[::-1]

        ods_pred = int(classes[order[0]])
        ods_trust = proba[order[0]]

        st.subheader("Resultado")
        st.success(
            f"**ODS {ods_pred} — {ODS_NAMES[ods_pred]}** "
            f"(ods_trust: {ods_trust:.1%})"
        )

        st.write("Probabilidades por ODS (top 5):")
        top5_idx = order[:5]
        resultados_df = pd.DataFrame({
            "ODS": [f"{int(classes[i])} - {ODS_NAMES[int(classes[i])]}" for i in top5_idx],
            "Probabilidad": [proba[i] for i in top5_idx],
        }).set_index("ODS")
        st.bar_chart(resultados_df)

st.caption(
    "MAIA UniAndes · Machine Learning No Supervisado · MicroProyecto 2 · Grupo 30"
)
