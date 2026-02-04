import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, legal, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch


# ==========================================================
# Page config + PIN lock
# ==========================================================
st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
TZ_NAME = str(st.secrets.get("BP_TZ", "")).strip()  # e.g. "America/Toronto"


def now_local() -> datetime:
    try:
        if TZ_NAME:
            return datetime.now(tz=ZoneInfo(TZ_NAME))
    except Exception:
        pass
    return datetime.now().astimezone()


if "bp_unlocked" not in st.session_state:
    st.session_state.bp_unlocked = False

with st.sidebar:
    st.markdown("### 🔒 Blisterpack Tracker Lock")
    pin_in = st.text_input("Enter PIN", type="password", placeholder="PIN")
    c_unlock, c_lock = st.columns(2)
    if c_unlock.button("Unlock", use_container_width=True):
        if pin_in == PIN_VALUE:
            st.session_state.bp_unlocked = True
            st.success("Unlocked.")
        else:
            st.session_state.bp_unlocked = False
            st.error("Wrong PIN.")
    if c_lock.button("Lock", use_container_width=True):
        st.session_state.bp_unlocked = False
        st.info("Locked.")

if not st.session_state.bp_unlocked:
    st.title("Daily Delivery Sheet")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


# ==========================================================
# DB setup (same DB as tracker)
# ==========================================================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA foreign_keys=ON;")
    return c


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
            FROM bp_patients
            WHERE active = 1
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["address"] = df["address"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df


def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    weeks_between = (d - anchor).days // 7
    return (weeks_between % interval_weeks) == 0


def read_overrides_for_date(d: date) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, odate, patient_name, action, packs, note
            FROM bp_overrides
            WHERE odate = ?
            ORDER BY id ASC
            """,
            c,
            params=(d.isoformat(),),
        )
    if df.empty:
        return df
    df["note"] = df["note"].fillna("")
    return df


@dataclass
class DayItem:
    patient: str
    address: str
    note: str
    packs: int
    interval_weeks: int


def build_day_items(d: date, patients_df: pd.DataFrame) -> list[DayItem]:
    if patients_df is None or patients_df.empty:
        return []
    todays = patients_df[patients_df["weekday"] == d.weekday()]
    out: list[DayItem] = []
    for _, r in todays.iterrows():
        anchor = r["anchor_date"]
        interval = int(r["interval_weeks"])
        if occurs_on_day(anchor, interval, d):
            out.append(
                DayItem(
                    patient=str(r["name"]),
                    address=str(r.get("address", "") or ""),
                    note=str(r.get("notes", "") or ""),
                    packs=int(r["packs_per_delivery"]),
                    interval_weeks=interval,
                )
            )
    out.sort(key=lambda x: (x.interval_weeks, x.patient.lower()))
    return out


def apply_overrides_day(items: list[DayItem], overrides_df: pd.DataFrame) -> list[DayItem]:
    if overrides_df is None or overrides_df.empty:
        return items

    out = items[:]
    for _, r in overrides_df.iterrows():
        name = str(r["patient_name"])
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])
        note = str(r.get("note", "") or "")

        if action == "skip":
            out = [x for x in out if x.patient != name]
        elif action == "add":
            out.append(
                DayItem(
                    patient=name,
                    address="",
                    note=note,
                    packs=int(packs or 1),
                    interval_weeks=99,
                )
            )

    out.sort(key=lambda x: (x.interval_weeks, x.patient.lower()))
    return out


# ==========================================================
# PDF helpers
# ==========================================================
def truncate_to_width(text: str, max_width: float, font_name: str, font_size: int) -> str:
    if not text:
        return ""
    if pdfmetrics.stringWidth(text, font_name, font_size) <= max_width:
        return text
    ell = "…"
    ell_w = pdfmetrics.stringWidth(ell, font_name, font_size)
    if ell_w >= max_width:
        return ell
    avail = max_width - ell_w
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if pdfmetrics.stringWidth(text[:mid], font_name, font_size) <= avail:
            lo = mid + 1
        else:
            hi = mid
    cut = max(0, lo - 1)
    return text[:cut].rstrip() + ell


def make_daily_sheet_pdf(d: date, items: list[DayItem], blank_rows: int, page_mode: str = "letter") -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize

    tmp_path = os.path.join("data", "_tmp_daily_sheet.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.45 * inch
    left = margin
    right = w - margin
    top = h - margin
    bottom = margin

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2.0, top, "SDM DELIVERY SHEET LOG")

    subtitle = f"{calendar.day_name[d.weekday()]} — {d.isoformat()}"
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2.0, top - 0.28 * inch, subtitle)

    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.28 * inch, f"Generated: {now_local().strftime('%Y-%m-%d %H:%M')}")

    # Table area
    table_top = top - 0.65 * inch
    table_bottom = bottom
    table_w = right - left
    table_h = table_top - table_bottom

    # Columns: Type, Patient, Address, Notes, Packages, Charged
    col_fracs = [0.10, 0.23, 0.38, 0.14, 0.075, 0.075]  # sums to 1.0 (Packages/Charged wider)
    col_w = [table_w * f for f in col_fracs]
    x = [left]
    for cw in col_w:
        x.append(x[-1] + cw)

    header_h = 0.32 * inch
    min_row_h = 0.24 * inch

    auto_rows = len(items)
    requested = auto_rows + int(blank_rows)

    max_fit = int((table_h - header_h) // min_row_h)
    max_fit = max(max_fit, 1)

    rows_fit = min(requested, max_fit)
    if auto_rows > rows_fit:
        rows_fit = max_fit  # still one page, but will truncate if insane

    row_h = (table_h - header_h) / rows_fit

    # Header background
    c.setFillGray(0.92)
    c.rect(left, table_top - header_h, table_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # Border + grid
    c.setStrokeGray(0.65)
    c.rect(left, table_bottom, table_w, table_h, stroke=1, fill=0)
    for xi in x[1:-1]:
        c.line(xi, table_bottom, xi, table_top)
    c.line(left, table_top - header_h, right, table_top - header_h)

    headers = ["Type", "Patient", "Address", "Notes", "Packages", "Charged"]
    c.setFont("Helvetica-Bold", 10)
    for i, htxt in enumerate(headers):
        c.drawString(x[i] + 6, table_top - header_h + 0.11 * inch, htxt)

    # Build rows: auto rows then BLANK rows (NOT RX)
    rows = []
    for it in items:
        rows.append([f"{int(it.packs)} BP", it.patient, it.address, it.note, "", ""])
    blanks_to_add = max(0, requested - len(rows))
    for _ in range(blanks_to_add):
        rows.append(["", "", "", "", "", ""])
    rows = rows[:rows_fit]

    # Content
    font_name = "Helvetica"
    fs = 10
    pad = 6
    c.setFont(font_name, fs)

    y0 = table_top - header_h
    for r_idx, row in enumerate(rows):
        y_top = y0 - r_idx * row_h
        y_bot = y_top - row_h

        c.line(left, y_bot, right, y_bot)

        text_y = y_top - 0.70 * row_h
        for ci, cell in enumerate(row):
            max_w = col_w[ci] - 2 * pad
            txt = truncate_to_width(str(cell), max_w, font_name, fs)
            c.drawString(x[ci] + pad, text_y, txt)

    c.showPage()
    c.save()

    with open(tmp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return data


# ==========================================================
# UI
# ==========================================================
st.title("Daily Delivery Sheet (auto BP + blank lines)")
st.caption("Pick a date → auto-fills scheduled BP → adds blank lines for anything extra (RX / manual).")

d = st.date_input("Pick day to print", value=date.today())
page_mode = st.selectbox("Paper size", ["letter", "legal"], index=0)
blank_rows = st.slider("Extra blank rows (for manual entries)", min_value=0, max_value=40, value=18)

patients_df = read_patients()
items = build_day_items(d, patients_df)
ov = read_overrides_for_date(d)
items = apply_overrides_day(items, ov)

st.subheader("Auto-filled BP deliveries for this day")
if not items:
    st.info("No scheduled BP deliveries for this day (or all filtered out). You can still print a blank sheet.")
else:
    preview = pd.DataFrame(
        {
            "Type": [f"{int(x.packs)} BP" for x in items],
            "Patient": [x.patient for x in items],
            "Address": [x.address for x in items],
            "Notes": [x.note for x in items],
        }
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)

pdf = make_daily_sheet_pdf(d=d, items=items, blank_rows=int(blank_rows), page_mode=page_mode)

st.download_button(
    "Download PDF — SDM DELIVERY SHEET LOG (one page)",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{d.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
