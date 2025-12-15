import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_DATA_PATHS = [
    "data/sdm_medications_enriched_offline.xlsx",
    "data/medications_enriched.xlsx",
    "data/medications_enriched.csv",
    "data/sdm_medications_enriched_offline.csv",
]

SEARCH_COLUMNS = ["brand_name", "DIN", "generic_name", "category", "form", "synonyms"]

# Where YOU will store images by brand_name
ASSETS_BY_NAME_DIR = Path("assets/meds_by_name")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# Phrase -> extra tokens (these tokens get added to the search query)
PHRASE_MAP: Dict[str, List[str]] = {
    # inhalers
    "blue inhaler": ["salbutamol", "ventolin", "rescue", "reliever", "puffer", "hfa", "inhaler"],
    "rescue inhaler": ["salbutamol", "ventolin", "puffer", "hfa", "inhaler"],
    "puffer": ["inhaler", "salbutamol", "symbicort"],

    # diuretics
    "water pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone", "swelling", "fluid"],
    "fluid pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone"],

    # diabetes injectables / pens
    "insulin pen": ["pen", "injection", "insulin", "ozempic", "wegovy"],
    "injection pen": ["pen", "injection", "ozempic", "wegovy"],

    # nasal sprays
    "nasal spray": ["spray", "nasal"],
}

# single token expansions
TOKEN_MAP: Dict[str, List[str]] = {
    "bp": ["blood pressure", "hypertension"],
    "uti": ["urinary", "infection"],
    "cholesterol": ["statin", "lipid"],
    "blood": ["anticoagulant", "clot", "stroke"],
    "allergy": ["hay fever", "antihistamine"],
}


# -----------------------------
# Helpers
# -----------------------------
def _find_data_path() -> str:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            return p
    envp = os.environ.get("SDM_DATA_PATH", "").strip()
    if envp and os.path.exists(envp):
        return envp
    return ""


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not path:
        raise FileNotFoundError(
            "Could not find a data file.\n\n"
            "Put your file in one of:\n"
            "- data/sdm_medications_enriched_offline.xlsx\n"
            "- data/medications_enriched.xlsx\n"
            "- data/medications_enriched.csv\n"
            "- data/sdm_medications_enriched_offline.csv\n\n"
            "or set SDM_DATA_PATH env var to your file path."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype={"DIN": str})
    else:
        # CSV: handle Windows encodings
        try:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="cp1252")

    # Ensure expected columns exist
    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Normalize NaNs/types
    for col in SEARCH_COLUMNS:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")

    # DIN: keep digits only (still useful for display/filtering)
    df["DIN"] = df["DIN"].apply(lambda x: re.sub(r"\D", "", str(x)))

    return df


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def expand_query(q: str) -> str:
    qn = normalize_text(q)
    extra_tokens: List[str] = []

    for phrase, adds in PHRASE_MAP.items():
        if phrase in qn:
            extra_tokens.extend(adds)

    tokens = re.split(r"[\s,/]+", qn)
    for t in tokens:
        if t in TOKEN_MAP:
            extra_tokens.extend(TOKEN_MAP[t])

    return " ".join([qn] + extra_tokens).strip()


def row_search_blob(row: pd.Series) -> str:
    parts = []
    for col in SEARCH_COLUMNS:
        parts.append(str(row.get(col, "")))
    return normalize_text(" | ".join(parts))


def matches_query(blob: str, expanded_query: str) -> bool:
    if not expanded_query:
        return True
    words = [w for w in re.split(r"[\s]+", expanded_query) if w]
    return all(w in blob for w in words)


def detect_missing(row: pd.Series) -> bool:
    return any(not str(row.get(c, "")).strip() for c in ["generic_name", "category", "form"])


def slugify(name: str) -> str:
    """
    Converts brand_name to a folder-safe slug.
    Must match YOUR folder names.
    """
    s = str(name or "").strip().lower()

    # fix common weird Excel mojibake
    s = s.replace("Ã©", "e").replace("â€", "").replace("â€™", "'").replace("Â", "")

    # only keep letters/numbers, everything else becomes "-"
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def find_images_for_brand_name(brand_name: str) -> dict:
    """
    Looks in assets/meds_by_name/<slug(brand_name)> for images.
    Returns {folder, box, pill, other}
    """
    folder = ASSETS_BY_NAME_DIR / slugify(brand_name)
    if not folder.exists():
        return {"folder": folder, "box": [], "pill": [], "other": []}

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    box = sorted([p for p in files if p.stem.lower().startswith("box")])
    pill = sorted([p for p in files if p.stem.lower().startswith(("pill", "tablet", "cap", "capsule"))])
    other = sorted([p for p in files if p not in box and p not in pill])

    return {"folder": folder, "box": box, "pill": pill, "other": other}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SDM Medication Navigator", layout="wide")

st.title("💊 SDM Medication Navigator")
st.caption("Offline lookup • Search by brand/generic/category/form/synonyms • Internal use")

data_path = _find_data_path()

with st.sidebar:
    st.header("Filters")

    st.write("**Data file**")
    st.code(data_path if data_path else "NOT FOUND", language="text")

    override = st.text_input("Override path (optional)", value="")
    if override.strip():
        data_path = override.strip()

    df = load_data(data_path)

    categories = ["All"] + sorted([c for c in df["category"].unique().tolist() if str(c).strip()])
    forms = ["All"] + sorted([f for f in df["form"].unique().tolist() if str(f).strip()])

    cat = st.selectbox("Category", categories, index=0)
    form = st.selectbox("Form", forms, index=0)

    st.divider()
    only_missing = st.checkbox("Show only rows missing info", value=False)

    st.divider()
    st.write("**Images folder**")
    st.code(str(ASSETS_BY_NAME_DIR), language="text")
    st.caption("Images are matched by folder name = slug(brand_name).")

search = st.text_input(
    "Search (brand / generic / category / form / synonyms)",
    placeholder="e.g. blue inhaler, water pill, nasal spray, blood thinner, apixaban",
)

expanded = expand_query(search)

# Apply filters
filtered = df.copy()

if cat != "All":
    filtered = filtered[filtered["category"].astype(str) == cat]

if form != "All":
    filtered = filtered[filtered["form"].astype(str) == form]

if only_missing:
    filtered = filtered[filtered.apply(detect_missing, axis=1)]

# Search
if expanded.strip():
    blobs = filtered.apply(row_search_blob, axis=1)
    mask = blobs.apply(lambda b: matches_query(b, expanded))
    filtered = filtered[mask]

st.subheader(f"Results ({len(filtered)})")

display_cols = ["brand_name", "generic_name", "DIN", "category", "form", "synonyms"]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

# Download filtered results
csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download current results as CSV",
    data=csv_bytes,
    file_name="sdm_medications_filtered.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Medication images")

if len(filtered) == 0:
    st.info("No results to preview. Search something first.")
else:
    # choose which result row to preview
    options = filtered["brand_name"].tolist()
    chosen = st.selectbox("Pick a medication to preview images", options)

    row = filtered[filtered["brand_name"] == chosen].iloc[0]

    st.write("**Selected**")
    st.write(
        f"- Brand: **{row.get('brand_name','')}**\n"
        f"- Generic: {row.get('generic_name','')}\n"
        f"- DIN: {row.get('DIN','')}\n"
        f"- Category: {row.get('category','')}\n"
        f"- Form: {row.get('form','')}\n"
    )

    imgs = find_images_for_brand_name(row.get("brand_name", ""))

    st.write("**Expected folder for images**")
    st.code(str(imgs["folder"]), language="text")

    if not (imgs["box"] or imgs["pill"] or imgs["other"]):
        st.info("No images found yet. Create the folder above and add box.jpg / pill.jpg etc.")
    else:
        # show box images first
        if imgs["box"]:
            st.write("**Box**")
            st.image([str(p) for p in imgs["box"]], use_container_width=True)

        # pills/tablets/capsules
        if imgs["pill"]:
            st.write("**Pill / Tablet / Capsule**")
            st.image([str(p) for p in imgs["pill"]], use_container_width=True)

        # everything else (device, spray, blister, etc.)
        if imgs["other"]:
            st.write("**Other**")
            st.image([str(p) for p in imgs["other"]], use_container_width=True)
