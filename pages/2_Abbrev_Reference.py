# pages/2_Abbrev_Reference.py

import streamlit as st
import pandas as pd

from sigcalc.engine import FREQ_DISPLAY, ROUTE_DISPLAY, TIMING_DISPLAY, FORM_DISPLAY

st.set_page_config(page_title="Abbreviation Reference", layout="centered")
st.title("Abbreviation Reference (Abbrev ↔ Full)")

query = st.text_input("Search (abbrev or meaning)", value="").strip().lower()

def make_df(title: str, d: dict):
    df = pd.DataFrame(
        [{"Abbrev": k, "Full form": v} for k, v in sorted(d.items(), key=lambda x: x[0])]
    )
    if query:
        mask = df["Abbrev"].str.lower().str.contains(query) | df["Full form"].str.lower().str.contains(query)
        df = df[mask]
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)

make_df("Frequency", FREQ_DISPLAY)
make_df("Routes", ROUTE_DISPLAY)
make_df("Timing", TIMING_DISPLAY)
make_df("Forms / Units", FORM_DISPLAY)

st.caption("Tip: type “twice” or “po” in search to filter quickly.")
