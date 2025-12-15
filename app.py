import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# OCR + image handling
from PIL import Image
import numpy as np

# OCR engine (EasyOCR)
import easyocr

# Fuzzy matching
from rapidfuzz import fuzz, process


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

# Where YOU store images by brand_name slug
ASSETS_BY_NAME_DIR = Path("assets/meds_by_name")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# OCR + matching thresholds (tweak if needed)
OCR_MIN_LINE_CHARS = 3
OCR_MATCH_THRESHOLD = 78         # match OCR line -> medication row
PROMPT_MATCH_THRESHOLD = 60      # match patient prompt -> meds found in screenshot

# Phrase -> extra tokens added to query
PHRASE_MAP: Dict[str, List[str]] = {
    "blue inhaler": ["salbutamol", "ventolin", "rescue", "reliever", "puffer", "hfa", "inhaler"],
    "rescue inhaler": ["salbutamol", "ventolin", "puffer", "hfa", "inhaler"],
    "puffer": ["inhaler", "salbutamol", "symbicort"],

    "water pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone", "swelling", "fluid"],
    "fluid pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone"],

    "insulin pen": ["pen", "injection", "insulin", "ozempic", "wegovy"],
    "injection pen": ["pen", "injection", "ozempic", "wegovy"],

    "nasal spray": ["spray", "nasal"],
}

TOKEN_MAP: Dict[str, List[str]] = {
    "bp": ["blood pressure", "hypertension"],
    "uti": ["urinary", "infection"],
    "cholesterol": ["statin", "lipid"],
    "blood": ["anticoagulant", "clot", "stroke"],
    "allergy": ["hay fever", "antihistamine"],
    "diabetes": ["metformin", "ozempic", "wegovy", "jardiance"],
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

    # DIN: keep digits only (still fine for display)
    if "DIN" in df.columns:
        df["DIN"] = df["DIN"].apply(lambda x: re.sub(r"\D", "", str(x)))

    # Keep brand_name as string
    df["brand_name"] = df["brand_name"].astype(str)

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


def matches_query_AND(blob: str, expanded_query: str) -> bool:
    """Simple AND match for normal search mode."""
    if not expanded_query:
        return True
    words = [w for w in re.split(r"[\s]+", expanded_query) if w]
    return all(w in blob for w in words)


def detect_missing(row: pd.Series) -> bool:
    return any(not str(row.get(c, "")).strip() for c in ["generic_name", "category", "form"])


def slugify(name: str) -> str:
    s = str(name or "").strip().lower()
    # fix common mojibake
    s = s.replace("Ã©", "e").replace("â€", "").replace("â€™", "'").replace("Â", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def find_images_for_brand_name(brand_name: str) -> dict:
    folder = ASSETS_BY_NAME_DIR / slugify(brand_name)
    if not folder.exists():
        return {"folder": folder, "box": [], "pill": [], "other": []}

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    box = sorted([p for p in files if p.stem.lower().startswith("box")])
    pill = sorted([p for p in files if p.stem.lower().startswith(("pill", "tablet", "cap", "capsule"))])
    other = sorted([p for p in files if p not in box and p not in pill])

    return {"folder": folder, "box": box, "pill": pill, "other": other}


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    # CPU is safest for Streamlit Cloud
    return easyocr.Reader(["en"], gpu=False)


def ocr_extract_lines(img: Image.Image) -> List[str]:
    reader = get_ocr_reader()
    arr = np.array(img.convert("RGB"))
    # easyocr returns list of (bbox, text, confidence)
    results = reader.readtext(arr)
    lines = []
    for _bbox, text, _conf in results:
        t = normalize_text(text)
        if len(t) >= OCR_MIN_LINE_CHARS:
            lines.append(t)
    # de-dup-ish
    lines = list(dict.fromkeys(lines))
    return lines


@st.cache_data(show_spinner=False)
def build_med_index(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Build a list of searchable strings (one per row) for fuzzy matching.
    Returns (choices, row_index_map)
    """
    choices = []
    row_ids = []
    for idx, row in df.iterrows():
        brand = str(row.get("brand_name", "")).strip()
        gen = str(row.get("generic_name", "")).strip()
        syn = str(row.get("synonyms", "")).strip()
        cat = str(row.get("category", "")).strip()
        form = str(row.get("form", "")).strip()
        din = str(row.get("DIN", "")).strip()
        blob = normalize_text(" | ".join([brand, gen, syn, cat, form, din]))
        choices.append(blob)
        row_ids.append(idx)
    return choices, row_ids


def match_screenshot_to_meds(df: pd.DataFrame, ocr_lines: List[str]) -> pd.DataFrame:
    """
    OCR lines -> find meds from your DB that likely appear in screenshot.
    Uses fuzzy matching against each row "blob".
    """
    if df.empty or not ocr_lines:
        return df.iloc[0:0].copy()

    choices, row_ids = build_med_index(df)

    found_scores: Dict[int, int] = {}

    for line in ocr_lines:
        # find best matches for this line
        hits = process.extract(
            query=line,
            choices=choices,
            scorer=fuzz.token_set_ratio,
            limit=8,
        )
        for choice_str, score, choice_pos in hits:
            if score >= OCR_MATCH_THRESHOLD:
                rid = row_ids[choice_pos]
                found_scores[rid] = max(found_scores.get(rid, 0), int(score))

    if not found_scores:
        return df.iloc[0:0].copy()

    out = df.loc[list(found_scores.keys())].copy()
    out["_ocr_score"] = out.index.map(lambda i: found_scores.get(i, 0))
    out = out.sort_values(["_ocr_score", "brand_name"], ascending=[False, True])
    return out


def match_prompt_within_found(found_df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """
    Patient prompt -> rank only within screenshot-detected meds.
    """
    if found_df.empty:
        return found_df

    expanded = expand_query(prompt)
    if not expanded.strip():
        return found_df

    # fuzzy match prompt against each found row blob
    ranked = []
    for idx, row in found_df.iterrows():
        blob = row_search_blob(row)
        score = fuzz.token_set_ratio(expanded, blob)
        ranked.append((idx, int(score)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    out = found_df.copy()
    out["_prompt_score"] = out.index.map(dict(ranked).get)
    out = out.sort_values(["_prompt_score", "_ocr_score", "brand_name"], ascending=[False, False, True])

    # Optional: hide low scores
    out = out[out["_prompt_score"] >= PROMPT_MATCH_THRESHOLD] if len(out) > 1 else out
    return out


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SDM Medication Navigator", layout="wide")

st.title("💊 SDM Medication Navigator")
st.caption("Search Mode + Screenshot Mode (OCR) • Internal use")

data_path = _find_data_path()

with st.sidebar:
    st.header("Data")

    st.write("**Data file detected**")
    st.code(data_path if data_path else "NOT FOUND", language="text")

    override = st.text_input("Override path (optional)", value="")
    if override.strip():
        data_path = override.strip()

    df = load_data(data_path)

    st.divider()
    st.header("Filters")
    categories = ["All"] + sorted([c for c in df["category"].unique().tolist() if str(c).strip()])
    forms = ["All"] + sorted([f for f in df["form"].unique().tolist() if str(f).strip()])

    cat = st.selectbox("Category", categories, index=0)
    form = st.selectbox("Form", forms, index=0)
    only_missing = st.checkbox("Show only rows missing info", value=False)

    st.divider()
    st.header("Assets")
    st.write("**Images folder**")
    st.code(str(ASSETS_BY_NAME_DIR), language="text")
    st.caption("Each med folder name = slug(brand_name). Example: assets/meds_by_name/alysena-28-100-20-mcg-4/box.jpg")

# Apply filters
base = df.copy()
if cat != "All":
    base = base[base["category"].astype(str) == cat]
if form != "All":
    base = base[base["form"].astype(str) == form]
if only_missing:
    base = base[base.apply(detect_missing, axis=1)]

tab_search, tab_screenshot = st.tabs(["🔎 Search Mode", "🖼️ Screenshot Mode (OCR)"])


# -----------------------------
# TAB 1: Normal Search Mode
# -----------------------------
with tab_search:
    search = st.text_input(
        "Search (brand / generic / category / form / synonyms)",
        placeholder="e.g. blue inhaler, water pill, nasal spray, blood thinner, apixaban",
        key="search_mode_input",
    )

    expanded = expand_query(search)
    filtered = base.copy()

    if expanded.strip():
        blobs = filtered.apply(row_search_blob, axis=1)
        mask = blobs.apply(lambda b: matches_query_AND(b, expanded))
        filtered = filtered[mask]

    st.subheader(f"Results ({len(filtered)})")

    display_cols = ["brand_name", "generic_name", "DIN", "category", "form", "synonyms"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    # Optional image preview here too
    st.divider()
    st.subheader("Preview images (optional)")

    if len(filtered) == 0:
        st.info("No results to preview.")
    else:
        chosen = st.selectbox("Pick a medication to preview images", filtered["brand_name"].tolist(), key="preview_search")
        row = filtered[filtered["brand_name"] == chosen].iloc[0]
        imgs = find_images_for_brand_name(row.get("brand_name", ""))

        st.caption("Expected folder:")
        st.code(str(imgs["folder"]), language="text")

        # Side-by-side (Box vs Pill) if available
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Box**")
            if imgs["box"]:
                st.image(str(imgs["box"][0]), use_container_width=True)
            else:
                st.caption("No box image found (box.jpg).")

        with col2:
            st.write("**Pill**")
            if imgs["pill"]:
                st.image(str(imgs["pill"][0]), use_container_width=True)
            else:
                st.caption("No pill image found (pill.jpg).")


# -----------------------------
# TAB 2: Screenshot Mode (OCR)
# -----------------------------
with tab_screenshot:
    st.write(
        "Upload a screenshot of the patient’s **current medication list** (from HealthWatch). "
        "Then type what the patient is asking for (e.g., **diabetes med**, **blue inhaler**, **water pill**)."
    )

    uploaded = st.file_uploader(
        "Upload screenshot (PNG/JPG/WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    patient_prompt = st.text_input(
        "Patient request (what are they asking for?)",
        placeholder="e.g. diabetes medication / blue inhaler / water pill / cholesterol pill",
        key="patient_prompt",
    )

    if uploaded is None:
        st.info("Upload a screenshot to begin.")
    else:
        img = Image.open(uploaded)
        st.image(img, caption="Screenshot uploaded", use_container_width=True)

        with st.spinner("Running OCR on screenshot..."):
            lines = ocr_extract_lines(img)

        with st.expander("OCR text detected (debug)"):
            if not lines:
                st.write("No OCR text found.")
            else:
                st.write(lines)

        with st.spinner("Matching OCR text to your medication list..."):
            found = match_screenshot_to_meds(base, lines)

        st.subheader(f"Meds detected from screenshot ({len(found)})")
        if found.empty:
            st.warning(
                "Couldn’t confidently match meds from the screenshot.\n\n"
                "Try:\n"
                "- higher resolution screenshot\n"
                "- include med names clearly\n"
                "- avoid tiny fonts\n"
            )
        else:
            show_cols = ["brand_name", "generic_name", "DIN", "category", "form", "synonyms", "_ocr_score"]
            show_cols = [c for c in show_cols if c in found.columns]
            st.dataframe(found[show_cols], use_container_width=True, hide_index=True)

        if not found.empty and patient_prompt.strip():
            ranked = match_prompt_within_found(found, patient_prompt)

            st.divider()
            st.subheader("Best matches for patient request")
            show_cols2 = ["brand_name", "generic_name", "DIN", "category", "form", "synonyms", "_ocr_score", "_prompt_score"]
            show_cols2 = [c for c in show_cols2 if c in ranked.columns]
            st.dataframe(ranked[show_cols2], use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Top pick + images")

            top = ranked.iloc[0]
            imgs = find_images_for_brand_name(top.get("brand_name", ""))

            st.write(
                f"**Top match:** {top.get('brand_name','')}\n\n"
                f"- Generic: {top.get('generic_name','')}\n"
                f"- Category: {top.get('category','')}\n"
                f"- Form: {top.get('form','')}\n"
                f"- OCR score: {top.get('_ocr_score','')}\n"
                f"- Prompt score: {top.get('_prompt_score','')}\n"
            )

            st.caption("Expected folder:")
            st.code(str(imgs["folder"]), language="text")

            # Side-by-side Box vs Pill
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Box**")
                if imgs["box"]:
                    st.image(str(imgs["box"][0]), use_container_width=True)
                else:
                    st.caption("No box image found (box.jpg).")
            with c2:
                st.write("**Pill**")
                if imgs["pill"]:
                    st.image(str(imgs["pill"][0]), use_container_width=True)
                else:
                    st.caption("No pill image found (pill.jpg).")
