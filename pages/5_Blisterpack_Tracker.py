import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, legal, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# Config + PIN
# =========================
st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
TZ_NAME = str(st.secrets.get("BP_TZ", "America/Toronto"))


def now_local() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(TZ_NAME))
    except Exception:
        return datetime.now()


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
    st.title("SDM DELIVERY SHEET LOG")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


# =========================
# SQLite (same DB as tracker)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def table_has_column(c: sqlite3.Connection, table: str, col: str) -> bool:
    rows = c.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def ensure_schema():
    with conn() as c:
        # Ensure base table exists
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT DEFAULT '',
                weekday INTEGER NOT NULL,
                interval_weeks INTEGER NOT NULL,
                packs_per_delivery INTEGER NOT NULL,
                anchor_date TEXT NOT NULL,
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        # Migrate address if old DB
        if not table_has_column(c, "bp_patients", "address"):
            c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT DEFAULT ''")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,
                packs INTEGER,
                note TEXT
            )
            """
        )
        c.commit()


ensure_schema()


# =========================
# Read data
# =========================
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
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["address"] = df["address"].fillna("").astype(str)
    return df


def month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start, end


def read_overrides(month_start: date, month_end: date) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, odate, patient_name, action, packs, note
            FROM bp_overrides
            WHERE odate >= ? AND odate <= ?
            """,
            c,
            params=(month_start.isoformat(), month_end.isoformat()),
        )
    if df.empty:
        return df
    df["odate"] = pd.to_datetime(df["odate"]).dt.date
    return df


# =========================
# Scheduling logic
# =========================
def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    weeks_between = ((d - anchor).days) // 7
    return (weeks_between % interval_weeks) == 0


@dataclass
class DeliveryItem:
    name: str
    packs: int
    interval_weeks: int
    address: str


def build_day_list(d: date, patients_df: pd.DataFrame, overrides_df: pd.DataFrame) -> list[DeliveryItem]:
    """Auto BP deliveries for a single day (weekdays only). Overrides can add/remove."""
    out: list[DeliveryItem] = []

    if patients_df.empty:
        return out

    # auto schedule weekdays only
    if d.weekday() <= 4:
        todays = patients_df[patients_df["weekday"] == d.weekday()]
        for _, r in todays.iterrows():
            if occurs_on_day(r["anchor_date"], int(r["interval_weeks"]), d):
                out.append(
                    DeliveryItem(
                        name=str(r["name"]),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=int(r["interval_weeks"]),
                        address=str(r.get("address", "") or ""),
                    )
                )

    # apply overrides for that date
    if not overrides_df.empty:
        day_ov = overrides_df[overrides_df["odate"] == d]
        for _, r in day_ov.iterrows():
            pname = str(r["patient_name"])
            action = str(r["action"]).strip().lower()
            packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])
            if action == "skip":
                out = [x for x in out if x.name != pname]
            elif action == "add":
                out.append(
                    DeliveryItem(
                        name=pname,
                        packs=packs or 1,
                        interval_weeks=99,
                        address="",
                    )
                )

    # order: weekly -> biweekly -> monthly -> manual
    out.sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return out


# =========================
# PDF helpers
# =========================
def truncate_to_width(text: str, max_width: float, font_name: str, font_size: int) -> str:
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


def bp_type_label(item: DeliveryItem) -> str:
    # Scheduled BP rows: show "1 BP / 2 BP / 4 BP"
    if item.packs in (1, 2, 4):
        return f"{item.packs} BP"
    return "BP"


def make_daily_pdf(
    day: date,
    bp_items: list[DeliveryItem],
    extra_blank_rows: int = 12,
    page_mode: str = "letter",
) -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize

    tmp_path = os.path.join(DATA_DIR, "_tmp_daily.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.35 * inch
    left = margin
    right = w - margin
    top = h - margin
    bottom = margin

    # Header (centered)
    title = "SDM DELIVERY SHEET LOG"
    sub = f"{day.strftime('%A')} — {day.isoformat()}"

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString((left + right) / 2, top, title)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString((left + right) / 2, top - 18, sub)

    # Generated time (top right)
    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 18, f"Generated: {now_local().strftime('%Y-%m-%d %H:%M')}")

    # Table geometry
    table_top = top - 40
    table_bottom = bottom
    table_h = table_top - table_bottom

    # Column widths (Notes smaller; Packages/Charged wider so they don't look weird)
    total_w = right - left

    # base widths in inches (then scaled)
    col_defs = [
        ("Type", 1.05),     # Type (1 BP / 2 BP / 4 BP)
        ("Patient", 2.40),
        ("Address", 4.30),
        ("Notes", 1.35),    # smaller notes
        ("Packages", 0.95), # wider than before
        ("Charged", 0.95),  # wider than before
    ]
    inch_total = sum(x[1] for x in col_defs)
    scale = total_w / (inch_total * inch)
    col_w = [x[1] * inch * scale for x in col_defs]
    headers = [x[0] for x in col_defs]

    # Rows: auto BP items + blank lines
    rows = []
    for it in bp_items:
        rows.append({
            "Type": bp_type_label(it),
            "Patient": it.name,
            "Address": it.address,
            "Notes": "",       # blank
            "Packages": "",    # blank
            "Charged": "",     # blank
        })
    for _ in range(extra_blank_rows):
        rows.append({"Type": "", "Patient": "", "Address": "", "Notes": "", "Packages": "", "Charged": ""})

    # Choose row height to fit page
    header_h = 0.32 * inch
    row_h = 0.30 * inch
    max_rows_fit = int((table_h - header_h) // row_h)
    if len(rows) > max_rows_fit:
        rows = rows[:max_rows_fit]  # hard cap (still one page)

    # Draw header background
    c.setFillGray(0.92)
    c.rect(left, table_top - header_h, sum(col_w), header_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # Draw grid + text
    x_positions = [left]
    for wcol in col_w:
        x_positions.append(x_positions[-1] + wcol)

    # Outer border
    c.setStrokeGray(0.65)
    c.rect(left, table_bottom, sum(col_w), header_h + row_h * len(rows), stroke=1, fill=0)

    # Vertical lines
    for x in x_positions:
        c.line(x, table_bottom, x, table_top)

    # Header text
    c.setFont("Helvetica-Bold", 9.5)
    y_header = table_top - header_h + 9
    for i, hname in enumerate(headers):
        c.drawString(x_positions[i] + 4, y_header, hname)

    # Horizontal line under header
    c.line(left, table_top - header_h, left + sum(col_w), table_top - header_h)

    # Rows
    c.setFont("Helvetica", 9)
    y = table_top - header_h
    for r in rows:
        y2 = y - row_h
        c.line(left, y2, left + sum(col_w), y2)

        # cell text
        pad = 4
        for i, key in enumerate(headers):
            cell_w = col_w[i] - (pad * 2)
            val = str(r.get(key, "") or "")
            val = truncate_to_width(val, cell_w, "Helvetica", 9)
            c.drawString(x_positions[i] + pad, y2 + 9, val)

        y = y2

    c.showPage()
    c.save()

    with open(tmp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return data


# =========================
# UI
# =========================
st.title("Daily Delivery Sheet (BP + manual extras)")

pick_day = st.date_input("Pick day to print", value=date.today())
paper = st.selectbox("Paper size", ["letter", "legal"], index=0)
extra_rows = st.slider("Extra blank lines (for RX / extra deliveries)", 5, 40, 18)

patients = read_patients()
mstart, mend = month_bounds(pick_day.year, pick_day.month)
overrides = read_overrides(mstart, mend)

bp_list = build_day_list(pick_day, patients, overrides)

st.markdown("### Auto-filled BP deliveries for this day")
if not bp_list:
    st.info("No BP deliveries found for this day (or it’s weekend). Overrides can still add manually.")
else:
    df_show = pd.DataFrame([{
        "Type": bp_type_label(it),
        "Patient": it.name,
        "Address": it.address
    } for it in bp_list])
    st.dataframe(df_show, use_container_width=True, hide_index=True)

pdf = make_daily_pdf(
    pick_day,
    bp_list,
    extra_blank_rows=int(extra_rows),
    page_mode=paper,
)

st.download_button(
    "Download Daily PDF — SDM DELIVERY SHEET LOG",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{pick_day.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
