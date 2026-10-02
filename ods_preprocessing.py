"""
Preprocesamiento de textos usado por el pipeline de clasificación ODS.

Este módulo se genera desde `MicroProyecto2G30.ipynb` (sección 1.4.4). La
función `text_preprocess` se define en un módulo aparte, y no dentro del
notebook, para que el pipeline exportado con `joblib` pueda cargarse desde
cualquier otro script (por ejemplo, la aplicación de Streamlit): `joblib`
guarda una referencia a `ods_preprocessing.text_preprocess`, no su código.
"""

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    _stopwords = stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords', quiet=True)
    _stopwords = stopwords.words('spanish')

tokenizer = RegexpTokenizer(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+')
spanish_stopwords = set(_stopwords)
stemmer = SnowballStemmer('spanish')


def text_preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token not in spanish_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)
