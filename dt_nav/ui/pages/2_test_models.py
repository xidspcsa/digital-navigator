import scipy.spatial
from gensim.models import Word2Vec
import streamlit as st
from annotated_text import annotated_text
from dt_nav.api import settings
from dt_nav.nlp.preprocess import CunningTokenizer
from dt_nav.nlp.w2v import embed_keyword
from dt_nav.processes.ner import extract_entities
from dt_nav.utils import jsonl_datum_to_annotated_text

tab_ner = st.tabs("NER")

with tab_ner:
    text = st.text_area("Enter text")

    if st.button("Extract"):
        tokenizer = CunningTokenizer()

        fixed_text = text
        # fixed_text = tokenizer.fix_punctuation(text)
        # fixed_text = tokenizer.fix_sentences(text)
        with st.expander("Fixed text"):
            st.write(fixed_text)

        datum = extract_entities(fixed_text)

        st.write(
            f'Total entities: {len(datum["entities"])}, unique: {len(datum["status"])}'
        )

        annotated_text(jsonl_datum_to_annotated_text(datum))
