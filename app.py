import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


# =============================
# Base paths (CRITICAL FIX)
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_DATA_PATHS = [
    DATA_DIR / "medications_enriched.csv",
    DATA_DIR / "medications.xlsx",
]

SEARCH_COLUMNS = ["brand_name", "DIN", "generic_name", "category", "form", "synonyms"]


# =============================
# Search helpers
# =============================
PHRASE_MAP: Dict[str, List[str]] = {
    "blue inhaler": ["salbutamol", "ventolin", "puffer", "rescue"],
    "water pill": ["diuretic", "furosemide", "hydrochlorothiazide"],
    "insulin pen": ["pen", "injection", "ozempic", "wegovy"],
    "nasal spray": ["spray", "nasal"],
}

TOKEN_MAP: Dict[str, List[str]] = {
    "bp": ["blood pressure", "hypertension"],
    "uti": ["urinary", "infection"],
    "cholesterol": ["statin", "lipid"],
    "blood": ["anticoagulant", "stroke"],
}


# =============================
# Helpers
# =============================
def find_data_path() -> Path | None:
    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".xlsx":
        df = pd.read_excel(path, dtype={"DIN": str})
    else:
        df = pd.read_csv(path, dtype={"DIN": str}, encoding="utf-8-sig")

    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in SEARCH_COLUMNS:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")

    df["DIN"] = df["DIN"].str.replace(r"\D", "", regex=True)

    return df


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def expand_query(q: str) -> str:
    qn = normalize_text(q)
    extra = []

    for phrase, tokens in PHRASE_MAP.items():
        if phrase in qn:
            extra.extend(tokens)

    for t in qn.split():
        if t in TOKEN_MAP:
            extra.extend(TOKEN_MAP[t])

    return " ".join([qn] + extra)


def row_blob(row: pd.Series) -> str:
    return normalize_text(" ".join(str(row[c]) for c in SEARCH_COLUMNS))


# =============================
# UI
# =============================
st.set_page_config(
    page_title="SDM Medication Navigator",
    layout="wide",
)

st.title("💊 SDM Medication Navigator")
st.caption("Offline medication lookup • Streamlit Cloud ready")

data_path = find_data_path()

with st.sidebar:
    st.header("Filters")

    if data_path:
        st.success(f"Data file loaded:\n{data_path.name}")
    else:
        st.error("Data file NOT FOUND")

    search = st.text_input(
        "Search medication",
        placeholder="e.g. amoxicillin, blue inhaler, water pill",
    )

if not data_path:
    st.stop()

df = load_data(data_path)

query = expand_query(search)

if query:
    mask = df.apply(lambda r: all(w in row_blob(r) for w in query.split()), axis=1)
    df = df[mask]

st.subheader(f"Results ({len(df)})")

st.dataframe(
    df[["brand_name", "generic_name", "DIN", "category", "form", "synonyms"]],
    use_container_width=True,
    hide_index=True,
)

csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download results as CSV",
    csv,
    "filtered_medications.csv",
    "text/csv",
)
