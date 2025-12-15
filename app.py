import os
import re
from pathlib import Path
from typing import Dict, List, Optional
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
# Local assets folder (can be committed if not huge)
ASSETS_BY_NAME_DIR = Path("assets/meds_by_name")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# Phrase -> extra tokens (added to query)
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
}
# -----------------------------
# Helpers
# -----------------------------
def repo_root() -> Path:
    # Streamlit Cloud runs from repo root; locally it also works.
    return Path(__file__).resolve().parent
def resolve_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str((repo_root() / p).resolve())
def _find_data_path() -> str:
    # allow env override first
    envp = os.environ.get("SDM_DATA_PATH", "").strip()
    if envp:
        ep = Path(envp)
        if ep.exists():
            return str(ep)
        # if env path is relative
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
        raise FileNotFoundError(
            "Could not find a data file.\n\n"
            "Put your file in one of:\n"
            "- data/sdm_medications_enriched_offline.xlsx\n"
            "- data/medications_enriched.xlsx\n"
            "- data/medications_enriched.csv\n"
            "- data/sdm_medications_enriched_offline.csv\n\n"
            "Or set SDM_DATA_PATH env var to your file path."
        )
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype={"DIN": str})
    else:
        # CSV: handle Windows encodings
        try:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype={"DIN": str}, encoding="cp1252")
    # ensure cols
    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # normalize NaNs/types
    for col in SEARCH_COLUMNS:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""}).fillna("")
    # DIN digits only (still useful)
    df["DIN"] = df["DIN"].apply(lambda x: re.sub(r"\D", "", str(x)))
    # trim brand names
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
    parts = []
    for col in SEARCH_COLUMNS:
        parts.append(str(row.get(col, "")))
    return normalize_text(" | ".join(parts))
def matches_query(blob: str, expanded_query: str) -> bool:
    if not expanded_query:
        return True
    words = [w for w in re.split(r"\s+", expanded_query) if w]
    return all(w in blob for w in words)
def detect_missing(row: pd.Series) -> bool:
    return any(not str(row.get(c, "")).strip() for c in ["generic_name", "category", "form"])
def slugify(name: str) -> str:
    """
    Converts brand_name to a folder-safe slug.
    Must match your folder names.
    """
    s = str(name or "").strip().lower()
    # common mojibake cleanup
    s = s.replace("Ã©", "e").replace("â€", "").replace("â€™", "'").replace("Â", "")
    s = s.replace("§", "").replace("’", "'")
    # only letters/numbers, others -> "-"
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s
def find_images_for_brand_name(brand_name: str) -> dict:
    """
    Looks in assets/meds_by_name/<slug(brand_name)> for images.
    Returns {folder, box, pill, other}
    """
    folder = (repo_root() / ASSETS_BY_NAME_DIR / slugify(brand_name)).resolve()
    if not folder.exists():
        return {"folder": folder, "box": [], "pill": [], "other": []}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    box = sorted([p for p in files if p.stem.lower().startswith("box")])
    pill = sorted([p for p in files if p.stem.lower().startswith(("pill", "tablet", "cap", "capsule"))])
    other = sorted([p for p in files if p not in box and p not in pill])
    return {"folder": folder, "box": box, "pill": pill, "other": other}
def safe_str(x) -> str:
    return "" if x is None else str(x)
def pick_first(paths: List[Path]) -> Optional[Path]:
    return paths[0] if paths else None
# -----------------------------
# UI (mobile-friendly)
# -----------------------------
st.set_page_config(
    page_title="SDM Medication Navigator",
    layout="wide",
    initial_sidebar_state="collapsed", # better on mobile
)

data_path = _find_data_path()
with st.sidebar:
    st.header("Settings")
    st.write("**Data file found**")
    st.code(data_path if data_path else "NOT FOUND", language="text")
    override = st.text_input("Override data path (optional)", value="", placeholder="e.g. data/medications_enriched.csv")
    if override.strip():
        data_path = resolve_path(override.strip())
    st.divider()
    st.write("**Images folder**")
    st.code(str((repo_root() / ASSETS_BY_NAME_DIR).resolve()), language="text")
    st.caption("Folder name must match slug(brand_name). Example: alysena-28-100-20-mcg-4")
df = load_data(data_path)

tabs = st.tabs(["Medication Search", "Patient Profile Analyzer"])

with tabs[0]:
    st.title("💊 SDM Medication Navigator")
    st.caption("Search by brand/generic/category/form/synonyms • Optional local images • Internal use")
    # compact controls row (better UX)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input(
            "Search",
            placeholder="e.g. diabetes, blue inhaler, water pill, apixaban, ozempic pen",
            label_visibility="visible",
        )
    with c2:
        only_missing = st.checkbox("Missing only", value=False)
    with c3:
        show_images = st.checkbox("Show images", value=True)
    expanded = expand_query(search)
    # filters in an expander (cleaner on mobile)
    with st.expander("Filters", expanded=False):
        categories = ["All"] + sorted([c for c in df["category"].unique().tolist() if safe_str(c).strip()])
        forms = ["All"] + sorted([f for f in df["form"].unique().tolist() if safe_str(f).strip()])
        fc1, fc2 = st.columns(2)
        with fc1:
            cat = st.selectbox("Category", categories, index=0)
        with fc2:
            form = st.selectbox("Form", forms, index=0)
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
    # download
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download current results as CSV",
        data=csv_bytes,
        file_name="sdm_medications_filtered.csv",
        mime="text/csv",
    )
    # -----------------------------
    # Autocomplete + images viewer
    # -----------------------------
    if show_images:
        st.divider()
        st.subheader("Quick preview")
        if len(filtered) == 0:
            st.info("No results to preview. Search something first.")
        else:
            # Autocomplete select: Streamlit selectbox already provides type-to-search
            # For mobile: keep it simple
            brand_list = filtered["brand_name"].astype(str).tolist()
            chosen = st.selectbox("Select a medication (type to search)", brand_list)
            row = filtered[filtered["brand_name"] == chosen].iloc[0]
            # info card
            info = f"""
**Brand:** {row.get('brand_name','')}
**Generic:** {row.get('generic_name','')}
**Category:** {row.get('category','')}
**Form:** {row.get('form','')}
**DIN:** {row.get('DIN','')}
"""
            st.markdown(info)
            imgs = find_images_for_brand_name(row.get("brand_name", ""))
            # folder help
            with st.expander("Where to put images for this medication", expanded=False):
                st.code(str(imgs["folder"]), language="text")
                st.caption("Add files like: box.jpg, pill.jpg (or .png/.webp).")
            if not (imgs["box"] or imgs["pill"] or imgs["other"]):
                st.warning("No images found for this medication yet.")
            else:
                # SIDE-BY-SIDE for box + pill
                box_img = pick_first(imgs["box"])
                pill_img = pick_first(imgs["pill"])
                if box_img or pill_img:
                    colA, colB = st.columns(2)
                    with colA:
                        st.write("**Box**")
                        if box_img:
                            st.image(str(box_img), use_container_width=True)
                        else:
                            st.caption("No box image.")
                    with colB:
                        st.write("**Pill / Tablet**")
                        if pill_img:
                            st.image(str(pill_img), use_container_width=True)
                        else:
                            st.caption("No pill image.")
                # show remaining images smaller (optional)
                remaining = []
                for p in imgs["box"][1:]:
                    remaining.append(p)
                for p in imgs["pill"][1:]:
                    remaining.append(p)
                for p in imgs["other"]:
                    remaining.append(p)
                if remaining:
                    with st.expander("More images", expanded=False):
                        st.image([str(p) for p in remaining], use_container_width=True)

with tabs[1]:
    st.header("Patient Profile Analyzer")
    st.caption("Upload a screenshot of the patient's Healthwatch profile and specify what you're looking for (e.g., 'diabetes medication'). The app will attempt to identify matching medications from the profile.")
    
    prompt = st.text_input(
        "What are you looking for?",
        placeholder="e.g., diabetes medication, cholesterol pill",
        label_visibility="visible",
    )
    
    uploaded_file = st.file_uploader(
        "Upload or drop screenshot here",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop or click to upload. Pasting directly isn't supported natively, but you can save the pasted image and upload it."
    )
    
    if uploaded_file and prompt:
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            st.error("""
            Required libraries not found. Please install them:
            
            pip install pillow pytesseract
            
            Also, install Tesseract OCR on your system:
            - On Ubuntu: sudo apt install tesseract-ocr
            - On macOS: brew install tesseract
            - On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
            - Then set the path if needed: pytesseract.pytesseract.tesseract_cmd = r'<path_to_tesseract.exe>'
            """)
            st.stop()
        
        # Optional: Set Tesseract path if needed (uncomment and adjust)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Example for macOS
        
        img = Image.open(uploaded_file)
        text = pytesseract.image_to_string(img).lower()
        
        with st.expander("Extracted text from screenshot (for debugging)", expanded=False):
            st.text(text)
        
        # Find medications mentioned in the text by matching against the dataframe
        found_meds = []
        for _, row in df.iterrows():
            brand_norm = normalize_text(row.get('brand_name', ''))
            generic_norm = normalize_text(row.get('generic_name', ''))
            if brand_norm in text or generic_norm in text:
                found_meds.append(row)
        
        if not found_meds:
            st.info("No medications from the database were detected in the screenshot.")
        else:
            found_df = pd.DataFrame(found_meds)
            
            # Expand the user's prompt and filter the found meds
            expanded_prompt = expand_query(prompt)
            if expanded_prompt.strip():
                blobs = found_df.apply(row_search_blob, axis=1)
                mask = blobs.apply(lambda b: matches_query(b, expanded_prompt))
                matching_df = found_df[mask]
            else:
                matching_df = found_df
            
            if matching_df.empty:
                st.info("No medications in the profile match your query.")
            else:
                st.subheader("Identified medication(s) that may need refill:")
                for _, med in matching_df.iterrows():
                    st.markdown(f"**{med.get('brand_name', 'Unknown')}** in the profile is the medication that needs refill.")
                    details = f"""
- **Generic:** {med.get('generic_name', '')}
- **Category:** {med.get('category', '')}
- **Form:** {med.get('form', '')}
- **DIN:** {med.get('DIN', '')}
- **Synonyms:** {med.get('synonyms', '')}
"""
                    st.markdown(details)
                
                st.dataframe(matching_df[display_cols], use_container_width=True, hide_index=True)
