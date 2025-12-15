import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

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
ASSETS_BY_NAME_DIR = Path("assets/meds_by_name")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Phrase expansions
PHRASE_MAP: Dict[str, List[str]] = {
    "blue inhaler": ["salbutamol", "ventolin", "rescue", "reliever", "puffer", "hfa", "inhaler"],
    "rescue inhaler": ["salbutamol", "ventolin", "puffer", "hfa", "inhaler"],
    "puffer": ["inhaler", "salbutamol", "symbicort"],
    "water pill": ["diuretic", "furosemide", "hydrochlorothiazide", "spironolactone"],
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
}

# -----------------------------
# Helpers
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parent

def resolve_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str((repo_root() / p).resolve())

def _find_data_path() -> str:
    envp = os.environ.get("SDM_DATA_PATH", "").strip()
    if envp:
        ep = Path(envp)
        if ep.exists():
            return str(ep)
        ep2 = Path(resolve_path(envp))
        if ep2.exists():
            return str(ep2)
    for p in DEFAULT_DATA_PATHS:
        rp = Path(resolve_path(p))
        if rp.exists():
            return str(rp)
    return ""

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not path:
        raise FileNotFoundError("Data file not found. Check paths or set SDM_DATA_PATH.")
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype={"DIN": str})
    else:
        try:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="cp1252")
    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in SEARCH_COLUMNS:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")
    df["DIN"] = df["DIN"].apply(lambda x: re.sub(r"\D", "", str(x)))
    df["brand_name"] = df["brand_name"].astype(str).str.strip()
    return df

def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def expand_query(q: str) -> str:
    qn = normalize_text(q)
    extra: List[str] = []
    for phrase, adds in PHRASE_MAP.items():
        if phrase in qn:
            extra.extend(adds)
    tokens = re.split(r"[\s,/]+", qn)
    for t in tokens:
        if t in TOKEN_MAP:
            extra.extend(TOKEN_MAP[t])
    return " ".join([qn] + extra).strip()

def row_search_blob(row: pd.Series) -> str:
    parts = [str(row.get(col, "")) for col in SEARCH_COLUMNS]
    return normalize_text(" | ".join(parts))

def matches_query(blob: str, expanded_query: str) -> bool:
    if not expanded_query:
        return True
    words = [w for w in re.split(r"\s+", expanded_query) if w]
    return all(w in blob for w in words)

def detect_missing(row: pd.Series) -> bool:
    return any(not str(row.get(c, "")).strip() for c in ["generic_name", "category", "form"])

def slugify(name: str) -> str:
    s = str(name or "").strip().lower()
    s = s.replace("Ã©", "e").replace("â€", "").replace("â€™", "'").replace("Â", "")
    s = s.replace("§", "").replace("’", "'")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def find_images_for_brand_name(brand_name: str) -> dict:
    folder = (repo_root() / ASSETS_BY_NAME_DIR / slugify(brand_name)).resolve()
    if not folder.exists():
        return {"folder": folder, "box": [], "pill": [], "other": []}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    box = sorted([p for p in files if p.stem.lower().startswith("box")])
    pill = sorted([p for p in files if p.stem.lower().startswith(("pill", "tablet", "cap", "capsule", "inhaler"))])
    other = sorted([p for p in files if p not in box and p not in pill])
    return {"folder": folder, "box": box, "pill": pill, "other": other}

def pick_first(paths: List[Path]) -> Optional[Path]:
    return paths[0] if paths else None

def safe_str(x) -> str:
    return "" if x is None else str(x)

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(
    page_title="SDM Medication Navigator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

data_path = _find_data_path()
with st.sidebar:
    st.header("Settings")
    st.write("**Data file**")
    st.code(data_path if data_path else "NOT FOUND", language="text")
    override = st.text_input("Override data path", placeholder="e.g. data/medications_enriched.csv")
    if override.strip():
        data_path = resolve_path(override.strip())
    st.divider()
    st.write("**Images folder**")
    st.code(str((repo_root() / ASSETS_BY_NAME_DIR).resolve()))

df = load_data(data_path)

tabs = st.tabs(["Medication Search", "Patient Profile Analyzer"])

# ==================== Medication Search Tab ====================
with tabs[0]:
    st.title("💊 SDM Medication Navigator")
    st.caption("Search by brand/generic/category/form/synonyms • Optional local images")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input("Search", placeholder="e.g. diabetes, blue inhaler, ozempic pen")
    with c2:
        only_missing = st.checkbox("Missing only", value=False)
    with c3:
        show_images = st.checkbox("Show images", value=True)

    expanded = expand_query(search)

    with st.expander("Filters", expanded=False):
        categories = ["All"] + sorted([c for c in df["category"].unique() if safe_str(c).strip()])
        forms = ["All"] + sorted([f for f in df["form"].unique() if safe_str(f).strip()])
        fc1, fc2 = st.columns(2)
        with fc1:
            cat = st.selectbox("Category", categories, index=0)
        with fc2:
            form = st.selectbox("Form", forms, index=0)

    filtered = df.copy()
    if cat != "All":
        filtered = filtered[filtered["category"].astype(str) == cat]
    if form != "All":
        filtered = filtered[filtered["form"].astype(str) == form]
    if only_missing:
        filtered = filtered[filtered.apply(detect_missing, axis=1)]
    if expanded.strip():
        blobs = filtered.apply(row_search_blob, axis=1)
        mask = blobs.apply(lambda b: matches_query(b, expanded))
        filtered = filtered[mask]

    st.subheader(f"Results ({len(filtered)})")
    display_cols = [c for c in ["brand_name", "generic_name", "DIN", "category", "form", "synonyms"] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download results as CSV",
        filtered[display_cols].to_csv(index=False).encode("utf-8-sig"),
        "sdm_medications_filtered.csv",
        "text/csv",
    )

    if show_images and len(filtered) > 0:
        st.divider()
        st.subheader("Quick preview")
        brand_list = filtered["brand_name"].astype(str).tolist()
        chosen = st.selectbox("Select medication", brand_list)
        row = filtered[filtered["brand_name"] == chosen].iloc[0]

        st.markdown(f"""
**Brand:** {row.get('brand_name','')}  
**Generic:** {row.get('generic_name','')}  
**Category:** {row.get('category','')}  
**Form:** {row.get('form','')}  
**DIN:** {row.get('DIN','')}
        """)

        imgs = find_images_for_brand_name(row.get("brand_name", ""))
        with st.expander("Image folder location"):
            st.code(str(imgs["folder"]))
            st.caption("Name files: box.jpg, pill.jpg, inhaler.jpg, etc.")

        if not (imgs["box"] or imgs["pill"] or imgs["other"]):
            st.info("No images found yet.")
        else:
            box_img = pick_first(imgs["box"])
            pill_img = pick_first(imgs["pill"])
            cols = st.columns(2)
            with cols[0]:
                st.write("**Box**")
                st.image(str(box_img) if box_img else "https://via.placeholder.com/300x200?text=No+Box+Image", use_container_width=True)
            with cols[1]:
                st.write("**Pill / Device**")
                st.image(str(pill_img) if pill_img else "https://via.placeholder.com/300x200?text=No+Pill+Image", use_container_width=True)

            remaining = imgs["box"][1:] + imgs["pill"][1:] + imgs["other"]
            if remaining:
                with st.expander("More images"):
                    st.image([str(p) for p in remaining], use_container_width=True)

# ==================== Patient Profile Analyzer Tab ====================
with tabs[1]:
    st.header("📸 Patient Profile Analyzer")
    st.caption("Upload a Healthwatch patient profile screenshot and tell us what you're looking for (e.g., 'blue inhaler', 'diabetes', 'cholesterol').")

    prompt = st.text_input(
        "What medication are you looking for?",
        placeholder="e.g., blue inhaler, diabetes medication, statin, blood pressure",
        key="analyzer_prompt"
    )

    uploaded_file = st.file_uploader(
        "Upload screenshot",
        type=["jpg", "jpeg", "png"],
        key="analyzer_upload"
    )

    if uploaded_file and prompt:
        try:
            import pytesseract
        except ImportError:
            st.error("Missing packages. Add to requirements.txt:\npillow\npytesseract")
            st.stop()

        with st.spinner("Reading text from image..."):
            img = Image.open(uploaded_file)
            img = img.convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(img, lang='eng+fra').lower()

        with st.expander("Extracted text (debug)", expanded=False):
            st.text(text)

        found_meds = []
        for _, row in df.iterrows():
            brand = normalize_text(row.get('brand_name', ''))
            generic = normalize_text(row.get('generic_name', ''))
            synonyms = normalize_text(row.get('synonyms', ''))
            if (brand and brand in text) or \
               (generic and generic in text) or \
               (synonyms and any(normalize_text(w) in text for w in synonyms.split() if len(w) > 3)):
                found_meds.append(row)

        if not found_meds:
            st.info("No medications from the database were found in the screenshot.")
            st.stop()

        found_df = pd.DataFrame(found_meds)
        expanded_prompt = expand_query(prompt)
        if expanded_prompt.strip():
            blobs = found_df.apply(row_search_blob, axis=1)
            mask = blobs.apply(lambda b: matches_query(b, expanded_prompt))
            matching_df = found_df[mask]
        else:
            matching_df = found_df

        if matching_df.empty:
            st.warning(f"No medication matches your request ('{prompt}'). All detected medications:")
            st.dataframe(found_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ Matching medication(s) found!")
            st.write("These medications appear in the profile and match your query:")

            for _, med in matching_df.iterrows():
                brand = med.get('brand_name', 'Unknown')
                st.markdown(f"### 💊 **{brand}** – likely needs refill")

                st.markdown(f"""
- **Generic:** {med.get('generic_name', '')}
- **Category:** {med.get('category', '')}
- **Form:** {med.get('form', '')}
- **DIN:** {med.get('DIN', '')}
- **Synonyms:** {med.get('synonyms', '')}
                """)

                imgs = find_images_for_brand_name(brand)
                if not (imgs["box"] or imgs["pill"] or imgs["other"]):
                    st.caption("No images available yet for this medication.")
                else:
                    box_img = pick_first(imgs["box"])
                    pill_img = pick_first(imgs["pill"])
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Box**")
                        if box_img:
                            st.image(str(box_img), use_container_width=True)
                        else:
                            st.caption("No box image")
                    with cols[1]:
                        st.write("**Pill / Device**")
                        if pill_img:
                            st.image(str(pill_img), use_container_width=True)
                        else:
                            st.caption("No pill/device image")

                    remaining = imgs["box"][1:] + imgs["pill"][1:] + imgs["other"]
                    if remaining:
                        with st.expander("More images"):
                            st.image([str(p) for p in remaining], use_container_width=True)

                st.divider()

            st.dataframe(matching_df[display_cols], use_container_width=True, hide_index=True)
