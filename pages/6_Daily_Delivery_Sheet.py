import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, legal
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics


# =========================
# Config + PIN lock
# =========================
st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
DEFAULT_TZ = str(st.secrets.get("BP_TZ", "UTC"))
if "bp_tz" not in st.session_state:
    st.session_state.bp_tz = DEFAULT_TZ

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT,
                weekday INTEGER NOT NULL,
                interval_weeks INTEGER NOT NULL,
                packs_per_delivery INTEGER NOT NULL,
                anchor_date TEXT NOT NULL,
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        # migration
        cols = [r[1] for r in c.execute("PRAGMA table_info(bp_patients)").fetchall()]
        if "address" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT")

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


init_db()


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

    st.divider()
    st.markdown("### 🕒 Timezone (for PDF stamp)")
    st.session_state.bp_tz = st.text_input("Timezone", value=st.session_state.bp_tz)

if not st.session_state.bp_unlocked:
    st.title("Daily Delivery Sheet (BP + manual extras)")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


WEEKDAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def now_local() -> datetime:
    try:
        return datetime.now(ZoneInfo(st.session_state.bp_tz))
    except Exception:
        return datetime.now(ZoneInfo("UTC"))


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
            FROM bp_patients
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    df["address"] = df["address"].fillna("")
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
    df["note"] = df["note"].fillna("")
    return df


def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    weeks_between = ((d - anchor).days // 7)
    return (weeks_between % interval_weeks) == 0


@dataclass
class DeliveryRow:
    type_label: str
    patient: str
    address: str
    notes: str


def build_day_rows(d: date, patients_df: pd.DataFrame, overrides_df: pd.DataFrame) -> list[DeliveryRow]:
    rows: list[DeliveryRow] = []
    if patients_df.empty:
        return rows

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return rows

    # Weekdays only for auto-schedule
    if d.weekday() <= 4:
        todays = active[active["weekday"] == d.weekday()]
        for _, r in todays.iterrows():
            anchor = r["anchor_date"]
            interval = int(r["interval_weeks"])
            if occurs_on_day(anchor, interval, d):
                packs = int(r["packs_per_delivery"])
                type_label = f"{packs} BP"
                rows.append(
                    DeliveryRow(
                        type_label=type_label,
                        patient=str(r["name"]),
                        address=str(r.get("address", "") or ""),
                        notes=str(r.get("notes", "") or ""),
                    )
                )

    # apply overrides for this exact day
    if not overrides_df.empty:
        day_ov = overrides_df[overrides_df["odate"] == d]
        for _, o in day_ov.iterrows():
            pname = str(o["patient_name"])
            action = str(o["action"]).lower().strip()
            packs = 1 if pd.isna(o.get("packs", None)) else int(o["packs"])
            note = str(o.get("note", "") or "")

            if action == "skip":
                rows = [x for x in rows if x.patient != pname]
            elif action == "add":
                # try address lookup by name
                addr = ""
                hit = active[active["name"] == pname]
                if not hit.empty:
                    addr = str(hit.iloc[0].get("address", "") or "")
                rows.append(DeliveryRow(type_label=f"{packs} BP", patient=pname, address=addr, notes=note))

    # Sort: 1 BP → 2 BP → 4 BP → others, then name
    def sort_key(r: DeliveryRow):
        # extract number at start if exists
        try:
            n = int(r.type_label.split()[0])
        except Exception:
            n = 99
        return (n, r.patient.lower())

    rows.sort(key=sort_key)
    return rows


def trunc(text: str, max_w: float, font: str, size: int) -> str:
    if pdfmetrics.stringWidth(text, font, size) <= max_w:
        return text
    ell = "…"
    avail = max_w - pdfmetrics.stringWidth(ell, font, size)
    if avail <= 1:
        return ell
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if pdfmetrics.stringWidth(text[:mid], font, size) <= avail:
            lo = mid + 1
        else:
            hi = mid
    return text[: max(0, lo - 1)].rstrip() + ell


def make_daily_pdf(d: date, rows: list[DeliveryRow], extra_blank_lines: int, paper: str = "letter") -> bytes:
    pagesize = letter if paper == "letter" else legal
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_daily.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.25 * inch
    left = margin
    right = w - margin
    top = h - margin
    bottom = margin

    stamp = now_local().strftime("%Y-%m-%d %H:%M")

    # Title (center)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, top - 0.25 * inch, "SDM DELIVERY SHEET LOG")

    # Subtitle (center)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, top - 0.48 * inch, f"{WEEKDAY_LABELS[d.weekday()]} — {d.isoformat()}")

    # Generated (right)
    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.48 * inch, f"Generated: {stamp}")

    # Table layout
    table_top = top - 0.75 * inch
    table_bottom = bottom
    table_h = table_top - table_bottom
    table_w = right - left

    # Column widths (sum = table_w)
    # Type | Patient | Address | Notes | Packages | Charged
    col_type = 0.75 * inch
    col_patient = 1.90 * inch
    col_address = 2.90 * inch
    col_notes = 0.95 * inch
    col_packages = 0.75 * inch
    col_charged = 0.75 * inch

    widths = [col_type, col_patient, col_address, col_notes, col_packages, col_charged]
    if abs(sum(widths) - table_w) > 2:
        # fallback scale if page size changes
        scale = table_w / sum(widths)
        widths = [x * scale for x in widths]

    headers = ["Type", "Patient", "Address", "Notes", "Packages", "Charged"]

    header_h = 0.32 * inch
    row_h = 0.28 * inch

    # compute max rows that fit on one page
    usable = table_h - header_h
    max_rows = int(usable // row_h)
    if max_rows < 5:
        max_rows = 5

    # final rows = bp rows + blank lines, capped
    all_rows = rows[:]
    blanks_needed = max(0, extra_blank_lines)
    # append blank lines (Type blank too)
    for _ in range(blanks_needed):
        all_rows.append(DeliveryRow(type_label="", patient="", address="", notes=""))

    all_rows = all_rows[:max_rows]

    # Header background
    c.setFillGray(0.92)
    c.rect(left, table_top - header_h, table_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # Grid coordinates
    x_positions = [left]
    for wcol in widths:
        x_positions.append(x_positions[-1] + wcol)

    # Draw header text + grid
    c.setStrokeGray(0.65)
    c.setFont("Helvetica-Bold", 9)

    for i, head in enumerate(headers):
        x0 = x_positions[i]
        x1 = x_positions[i + 1]
        c.drawString(x0 + 4, table_top - header_h + 8, head)
        c.line(x0, table_top, x0, table_bottom)
        c.line(x0, table_top - header_h, x1, table_top - header_h)

    c.line(right, table_top, right, table_bottom)
    c.line(left, table_top, right, table_top)

    # Rows
    c.setFont("Helvetica", 9)
    y = table_top - header_h

    for r in all_rows:
        y2 = y - row_h

        # horizontal line
        c.setStrokeGray(0.65)
        c.line(left, y2, right, y2)

        # cell text (truncate)
        pad = 4
        c.drawString(x_positions[0] + pad, y2 + 7, trunc(r.type_label, widths[0] - 2 * pad, "Helvetica", 9))
        c.drawString(x_positions[1] + pad, y2 + 7, trunc(r.patient, widths[1] - 2 * pad, "Helvetica", 9))
        c.drawString(x_positions[2] + pad, y2 + 7, trunc(r.address, widths[2] - 2 * pad, "Helvetica", 9))
        c.drawString(x_positions[3] + pad, y2 + 7, trunc(r.notes, widths[3] - 2 * pad, "Helvetica", 9))
        # Packages + Charged intentionally blank

        y = y2

    # bottom border
    c.line(left, table_bottom, right, table_bottom)

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
st.caption("Auto-fills BP patients due today, then prints blank lines for anything else you want to write manually.")

pick_day = st.date_input("Pick day to print", value=date.today())
paper = st.selectbox("Paper size", ["letter", "legal"], index=0)

# extra blank lines (capped to fit one page)
extra_lines = st.slider("Extra blank lines (for extra deliveries)", min_value=5, max_value=40, value=18)

patients = read_patients()
mstart, mend = month_bounds(pick_day.year, pick_day.month)
overrides = read_overrides(mstart, mend)

rows = build_day_rows(pick_day, patients, overrides)

st.subheader("Auto-filled BP deliveries for this day")
if rows:
    st.dataframe(pd.DataFrame([r.__dict__ for r in rows]), use_container_width=True, hide_index=True)
else:
    st.info("No BP deliveries due for this day (or it's weekend). You can still print blanks and write manually.")

pdf = make_daily_pdf(pick_day, rows, extra_blank_lines=int(extra_lines), paper=paper)

st.download_button(
    "Download Daily PDF (SDM DELIVERY SHEET LOG)",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{pick_day.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
