import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, legal, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# =========================
# Config (same PIN + DB)
# =========================
st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
TZ_NAME = str(st.secrets.get("BP_TZ", "America/Toronto"))

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

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
    st.title("Daily Delivery Sheet (BP + manual extras)")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# =========================
# Read patients (needs address for auto-fill)
# =========================
def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, COALESCE(address,'') AS address, weekday, interval_weeks, packs_per_delivery,
                   anchor_date, COALESCE(notes,'') AS notes, active
            FROM bp_patients
            WHERE active = 1
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
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
# Scheduling logic (same as tracker)
# =========================
def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    delta_days = (d - anchor).days
    weeks_between = delta_days // 7
    return (weeks_between % interval_weeks) == 0


@dataclass
class DeliveryItem:
    name: str
    packs: int
    interval_weeks: int  # 1/2/4 or 99 manual
    address: str = ""


def build_day_items(d: date, patients_df: pd.DataFrame, overrides_df: pd.DataFrame) -> list[DeliveryItem]:
    items: list[DeliveryItem] = []
    if patients_df.empty:
        patients_df = pd.DataFrame()

    # auto schedule weekdays only
    if d.weekday() <= 4 and not patients_df.empty:
        todays = patients_df[patients_df["weekday"] == d.weekday()]
        for _, r in todays.iterrows():
            if occurs_on_day(r["anchor_date"], int(r["interval_weeks"]), d):
                items.append(
                    DeliveryItem(
                        name=str(r["name"]),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=int(r["interval_weeks"]),
                        address=str(r.get("address", "") or ""),
                    )
                )

    # apply overrides for this date
    if overrides_df is not None and not overrides_df.empty:
        od = overrides_df[overrides_df["odate"] == d]
        for _, r in od.iterrows():
            nm = str(r["patient_name"])
            act = str(r["action"]).lower().strip()
            pk = 1 if pd.isna(r.get("packs", None)) else int(r["packs"])
            if act == "skip":
                items = [x for x in items if x.name != nm]
            elif act == "add":
                items.append(DeliveryItem(name=nm, packs=pk, interval_weeks=99, address=""))

    # order: weekly -> biweekly -> monthly -> manual, then name
    items.sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return items


# =========================
# Daily PDF (SDM DELIVERY SHEET LOG)
# =========================
def make_daily_sdm_pdf(d: date, day_items: list[DeliveryItem], page_mode: str, extra_blank_rows: int) -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize

    tz = ZoneInfo(TZ_NAME)
    generated = datetime.now(tz)

    tmp_path = os.path.join(DATA_DIR, "_tmp_daily_sdm.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    # margins
    margin = 0.45 * inch
    left = margin
    right = w - margin
    top = h - margin
    bottom = margin

    # Header centered
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, top, "SDM DELIVERY SHEET LOG")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, top - 0.25 * inch, f"{d.strftime('%A')} — {d.isoformat()}")

    # Generated stamp top-right (local timezone)
    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.25 * inch, f"Generated: {generated.strftime('%Y-%m-%d %H:%M')}")

    # Table geometry
    table_top = top - 0.55 * inch
    table_bottom = bottom
    table_h = table_top - table_bottom

    # Column widths (sum = usable width)
    usable_w = right - left  # ~10.0 in on landscape letter with these margins
    # Type, Patient, Address, Notes (smaller), Packages, Charged
    col_w = [0.9*inch, 2.1*inch, 3.6*inch, 1.4*inch, 1.0*inch, 1.0*inch]
    # If legal and more space, stretch Address a bit
    if page_mode == "legal":
        extra = (usable_w - sum(col_w))
        if extra > 0:
            col_w[2] += extra  # give extra to Address
    else:
        # clamp to usable width if needed (rare printer differences)
        if sum(col_w) > usable_w:
            scale = usable_w / sum(col_w)
            col_w = [cw * scale for cw in col_w]

    x = [left]
    for cw in col_w:
        x.append(x[-1] + cw)

    header_h = 0.32 * inch
    row_h = 0.30 * inch  # good handwriting space

    # Determine max rows that fit
    max_rows = int((table_h - header_h) // row_h)
    if max_rows < 1:
        max_rows = 1

    # Build rows: scheduled items first, then blank rows
    def type_label(item: DeliveryItem) -> str:
        # EXACT format user asked: "1 BP", "2 BP", "4 BP"
        return f"{int(item.packs)} BP"

    rows = []
    for it in day_items:
        rows.append({
            "Type": type_label(it),
            "Patient": it.name,
            "Address": it.address or "",
            "Notes": "",
            "Packages": "",  # leave empty to write
            "Charged": "",   # leave empty to write (cc/0)
        })

    # add blank rows (NO 'RX' text)
    blanks_needed = max(0, min(extra_blank_rows, max_rows - len(rows)))
    for _ in range(blanks_needed):
        rows.append({"Type": "", "Patient": "", "Address": "", "Notes": "", "Packages": "", "Charged": ""})

    # also fill remaining space up to max_rows (always give full grid)
    while len(rows) < max_rows:
        rows.append({"Type": "", "Patient": "", "Address": "", "Notes": "", "Packages": "", "Charged": ""})

    # Draw header row background
    c.setFillGray(0.92)
    c.rect(left, table_top - header_h, right - left, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # Header labels
    headers = ["Type", "Patient", "Address", "Notes", "Packages", "Charged"]
    c.setFont("Helvetica-Bold", 9)
    y_header = table_top - header_h + 0.10 * inch
    for i, hname in enumerate(headers):
        c.drawString(x[i] + 6, y_header, hname)

    # Grid lines
    c.setStrokeGray(0.65)
    # Outer border
    c.rect(left, table_top - header_h - max_rows * row_h, right - left, header_h + max_rows * row_h, stroke=1, fill=0)

    # Vertical lines
    for xi in x:
        c.line(xi, table_top - header_h - max_rows * row_h, xi, table_top)

    # Horizontal lines (header + each row)
    c.line(left, table_top - header_h, right, table_top - header_h)
    for r in range(max_rows + 1):
        yy = table_top - header_h - r * row_h
        c.line(left, yy, right, yy)

    # Cell text
    c.setFont("Helvetica", 9)
    for r, row in enumerate(rows[:max_rows]):
        yy = table_top - header_h - (r + 1) * row_h + 0.10 * inch

        c.drawString(x[0] + 6, yy, str(row["Type"])[:10])
        c.drawString(x[1] + 6, yy, str(row["Patient"])[:50])
        c.drawString(x[2] + 6, yy, str(row["Address"])[:70])
        c.drawString(x[3] + 6, yy, str(row["Notes"])[:30])
        # Packages + Charged intentionally blank (leave for writing)

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

pick = st.date_input("Pick day to print", value=date.today())
paper = st.selectbox("Paper size", ["letter", "legal"], index=0)

extra = st.slider(
    "Extra blank lines (for extra deliveries)",
    min_value=0,
    max_value=60,
    value=20,
)

patients_df = read_patients()
mstart, mend = month_bounds(pick.year, pick.month)
overrides_df = read_overrides(mstart, mend)

items = build_day_items(pick, patients_df, overrides_df)

st.subheader("Auto-filled BP deliveries for this day")
if not items:
    st.info("No BP deliveries found for this day (or it’s weekend and no overrides).")
else:
    preview = pd.DataFrame([{
        "Type": f"{it.packs} BP",
        "Patient": it.name,
        "Address": it.address or "",
    } for it in items])
    st.dataframe(preview, use_container_width=True, hide_index=True)

pdf = make_daily_sdm_pdf(pick, items, paper, extra)

st.download_button(
    "Download Daily PDF — SDM DELIVERY SHEET LOG (1 page)",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{pick.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
