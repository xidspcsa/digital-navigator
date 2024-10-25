import os
from itertools import count

import streamlit as st
from dt_nav.api import RedisConn
from dt_nav.api.redis_api import get_stored_vacancies_len, reset_stored_vacancies_ids
from dt_nav.processes.documents.common import mark_dupes
from dt_nav.processes.embeddings.refresh import refresh_keyword_embeddings
from dt_nav.processes.embeddings.train import train_w2v
from dt_nav.processes.integration.iot import sync_iot_plan_data, sync_iot_rpd_data
from dt_nav.processes.integration.vacancies import (
    DataCollectorHH,
    load_hh_ids,
    parse_hh,
    parse_rvr,
)
from dt_nav.processes.ner.process_documents import (
    extract_entities_for_document,
    extract_entities_for_document_type,
)
from dt_nav.processes.ner.train import train_ner
from dt_nav.processes.vacancy_clustering import calculate_cluster_keywords

RVR_SETS = "./data/rvr/"
HH_SETS = "./data/hh/"

BUTTON_KEY = (f"button-{c}" for c in count(0, 1))

ner_tab = st.tabs("NER")

with ner_tab:
    with st.expander("ner.train_ner"):
        if st.button("Send", key=next(BUTTON_KEY)):
            train_ner.send()

    with st.expander("ner.extract_entities_for_document"):
        needle = st.text_input("Needle", value="rpd:8366")
        update = st.selectbox("Update", options=[0, 1, 2], index=1)
        if st.button("Send", key=next(BUTTON_KEY)):
            extract_entities_for_document.send(needle=needle, update=update)

    with st.expander("ner.extract_entities_for_document_type"):
        object_type = st.text_input("Object type", value="rpd")
        update = st.selectbox(
            "Update",
            options=[0, 1, 2],
            index=1,
            key=next(BUTTON_KEY),
        )
        if st.button("Send", key=next(BUTTON_KEY)):
            extract_entities_for_document_type.send(type_=object_type, update=update)

#...