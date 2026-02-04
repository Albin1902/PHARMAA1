import os
import sqlite3
import calendar
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, legal, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics


# =========================
# Page config + PIN lock
# =========================
st.set_page_config(page_title="SDM Delivery Sheet Log", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))

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
# DB setup (same DB)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start, end


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
            FROM bp_patients
            ORDER BY active DESC, weekday ASC, interval_weeks ASC, name ASC
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    df["address"] = df["address"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df


def read_overrides(month_start: date, month_end: date) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, odate, patient_name, action, packs, note
            FROM bp_overrides
            WHERE odate >= ? AND odate <= ?
            ORDER BY odate ASC, patient_name ASC
            """,
            c,
            params=(month_start.isoformat(), month_end.isoformat()),
        )
    if df.empty:
        return df
    df["odate"] = pd.to_datetime(df["odate"]).dt.date
    df["note"] = df["note"].fillna("")
    return df


def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    delta_days = (d - anchor).days
    weeks_between = delta_days // 7
    return (weeks_between % interval_weeks) == 0


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[dict]]:
    # returns {date: [ {name,packs,interval,address}, ... ] }
    start, end = month_bounds(year, month)
    schedule = {}
    d = start
    while d <= end:
        schedule[d] = []
        d += timedelta(days=1)

    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # auto schedule weekdays only
        if d.weekday() > 4:
            continue

        todays = active[active["weekday"] == d.weekday()]
        if todays.empty:
            continue

        for _, r in todays.iterrows():
            anchor = r["anchor_date"]
            interval = int(r["interval_weeks"])
            if occurs_on_day(anchor, interval, d):
                schedule[d].append({
                    "name": str(r["name"]),
                    "packs": int(r["packs_per_delivery"]),
                    "interval": interval,
                    "address": str(r.get("address", "") or ""),
                })

        schedule[d].sort(key=lambda x: (x["packs"], x["name"].lower()))
    return schedule


def apply_overrides(schedule: dict[date, list[dict]], overrides_df: pd.DataFrame):
    if overrides_df.empty:
        return schedule
    for _, r in overrides_df.iterrows():
        d = r["odate"]
        name = str(r["patient_name"])
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x["name"] != name]
        elif action == "add":
            schedule[d].append({"name": name, "packs": packs or 1, "interval": 99, "address": ""})
            schedule[d].sort(key=lambda x: (x["packs"], x["name"].lower()))
    return schedule


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


def make_daily_pdf(
    delivery_date: date,
    rows: list[dict],
    page_mode: str = "letter",
    rx_rows: int = 8,
) -> bytes:
    """
    ONE PAGE daily sheet:
      Type | Patient | Address | Notes | Packages | Charged
    - Type values like: "1 BP", "2 BP", "4 BP", "RX"
    - Packages/Charged EMPTY for writing
    - Adds RX blank rows + extra blanks until page filled
    """
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_daily.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.35 * inch
    left = margin
    right = w - margin
    top = h - margin
    bottom = margin

    # Title centered
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2, top, "SDM DELIVERY SHEET LOG")

    c.setFont("Helvetica-Bold", 12)
    subtitle = f"{delivery_date.strftime('%A')} — {delivery_date.isoformat()}"
    c.drawCentredString(w / 2, top - 0.28 * inch, subtitle)

    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.28 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    table_top = top - 0.55 * inch
    table_bottom = bottom
    table_left = left
    table_right = right

    table_w = table_right - table_left
    table_h = table_top - table_bottom

    header_h = 0.30 * inch
    row_h = 0.30 * inch

    # Type, Patient, Address, Notes, Packages, Charged
    # Notes smaller; Packages/Charged wider so headers don't smash
    col_fracs = [0.10, 0.24, 0.43, 0.13, 0.05, 0.05]
    col_w = [table_w * f for f in col_fracs]

    headers = ["Type", "Patient", "Address", "Notes", "Packages", "Charged"]

    # calculate max rows that fit
    max_rows = int((table_h - header_h) // row_h)
    if max_rows < 1:
        max_rows = 1

    # Build final printable rows
    out_rows = []
    for r in rows:
        out_rows.append({
            "type": f'{int(r.get("packs", 1))} BP',
            "patient": str(r.get("name", "")),
            "address": str(r.get("address", "")),
            "notes": "",
            "packages": "",
            "charged": "",
        })

    # RX blanks
    for _ in range(rx_rows):
        out_rows.append({
            "type": "RX",
            "patient": "",
            "address": "",
            "notes": "",
            "packages": "",
            "charged": "",
        })

    # Fill remaining with blanks (no Type)
    while len(out_rows) < max_rows:
        out_rows.append({
            "type": "",
            "patient": "",
            "address": "",
            "notes": "",
            "packages": "",
            "charged": "",
        })

    out_rows = out_rows[:max_rows]

    # Header background
    c.setFillGray(0.92)
    c.rect(table_left, table_top - header_h, table_w, header_h, stroke=0, fill=1)
    c.setFillGray(0)

    # Header text (slightly smaller + centered on tiny cols)
    fs_header = 9
    c.setFont("Helvetica-Bold", fs_header)

    x = table_left
    for i, htxt in enumerate(headers):
        max_w = col_w[i] - 8
        label = truncate_to_width(htxt, max_w, "Helvetica-Bold", fs_header)

        if htxt in ("Packages", "Charged"):
            c.drawCentredString(x + (col_w[i] / 2), table_top - header_h + 9, label)
        else:
            c.drawString(x + 4, table_top - header_h + 9, label)

        x += col_w[i]

    # Grid
    c.setStrokeGray(0.70)

    # Outer border
    c.rect(table_left, table_top - header_h - (max_rows * row_h), table_w, header_h + (max_rows * row_h), stroke=1, fill=0)

    # Vertical lines
    x = table_left
    for wcol in col_w:
        c.line(x, table_top, x, table_top - header_h - (max_rows * row_h))
        x += wcol
    c.line(table_right, table_top, table_right, table_top - header_h - (max_rows * row_h))

    # Horizontal lines
    y = table_top - header_h
    c.line(table_left, y, table_right, y)
    for r in range(max_rows):
        y2 = y - row_h
        c.line(table_left, y2, table_right, y2)
        y = y2

    # Fill text
    fs = 9
    c.setFont("Helvetica", fs)

    y = table_top - header_h
    for r in range(max_rows):
        row = out_rows[r]
        y_text = y - row_h + 0.20 * inch

        cols = [
            row["type"],
            row["patient"],
            row["address"],
            row["notes"],
            row["packages"],
            row["charged"],
        ]

        x = table_left
        for i, txt in enumerate(cols):
            max_w = col_w[i] - 8
            line = truncate_to_width(str(txt), max_w, "Helvetica", fs)

            # center tiny cols
            if i in (4, 5):
                c.drawCentredString(x + col_w[i] / 2, y_text, line)
            else:
                c.drawString(x + 4, y_text, line)

            x += col_w[i]

        y -= row_h

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
st.title("SDM DELIVERY SHEET LOG")
st.caption("Auto-fills BP deliveries from your schedule + adds RX blank lines. One page, printable.")

today = date.today()

c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
with c1:
    dsel = st.date_input("Delivery date", value=today)
with c2:
    page_mode = st.selectbox("Paper size", ["letter", "legal"], index=0)
with c3:
    rx_rows = st.number_input("RX blank rows", min_value=0, max_value=40, value=8, step=1)

patients_df = read_patients()

# Build schedule for selected date’s month + overrides
base = build_month_schedule(dsel.year, dsel.month, patients_df)
ms, me = month_bounds(dsel.year, dsel.month)
ov = read_overrides(ms, me)
base = apply_overrides(base, ov)

rows = base.get(dsel, [])
rows = sorted(rows, key=lambda x: (x["packs"], x["name"].lower()))

st.subheader(f"{dsel.strftime('%A')} — {dsel.isoformat()}")
if not rows:
    st.info("No BP deliveries scheduled for this day (auto). PDF will still include RX + blank lines.")
else:
    st.dataframe(pd.DataFrame(rows)[["packs", "name", "address"]], use_container_width=True, hide_index=True)

pdf = make_daily_pdf(dsel, rows, page_mode=page_mode, rx_rows=int(rx_rows))

st.download_button(
    "Download Daily Sheet PDF (ONE PAGE)",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{dsel.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
