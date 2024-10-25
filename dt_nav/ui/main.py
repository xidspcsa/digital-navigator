from _common import fix_import

fix_import()

import pandas as pd
import streamlit as st
from dt_nav.api.health import run_healthchecks

st.subheader("Digital Trajectories Navigator 2: Electric Boogaloo")

health = run_healthchecks()

gut_count = len([v for v in health.values() if v["ok"] is True])
schlecht_count = len([v for v in health.values() if v["ok"] is False])
total_count = len(health)

if schlecht_count == 0:
    st.success(f"{gut_count}/{total_count} components work")
elif gut_count > 0:
    st.warning(f"{gut_count}/{total_count} components work")
else:
    st.error(f"{gut_count}/{total_count} components work :c")

df = pd.DataFrame(
    [{"component": key, "status": value["ok"]} for key, value in health.items()]
)
st.write(df)

if schlecht_count > 0:
    for key, value in health.items():
        if value['ok'] is True:
            continue
        with st.expander(f'Exception in {key}'):
            st.exception(value['exception'])
