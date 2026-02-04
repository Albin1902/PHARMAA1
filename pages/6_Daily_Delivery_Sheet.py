import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter, legal, landscape, portrait
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics


st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
TZ_NAME = str(st.secrets.get("BP_TZ", "America/Toronto"))
try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:
    TZ = ZoneInfo("UTC")


def now_local():
    return datetime.now(TZ)


DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def require_pin():
    if "bp_unlocked" not in st.session_state:
        st.session_state.bp_unlocked = False

    with st.sidebar:
        st.markdown("### 🔒 Blisterpack Tracker Lock")
        pin_in = st.text_input("Enter PIN", type="password", placeholder="PIN")
        c_unlock, c_lock = st.columns(2)
        if c_unlock.button("Unlock", use_container_width=True):
            st.session_state.bp_unlocked = (pin_in == PIN_VALUE)
            st.success("Unlocked.") if st.session_state.bp_unlocked else st.error("Wrong PIN.")
        if c_lock.button("Lock", use_container_width=True):
            st.session_state.bp_unlocked = False
            st.info("Locked.")

    if not st.session_state.bp_unlocked:
        st.title("Daily Delivery Sheet (BP + manual extras)")
        st.warning("Locked. Enter PIN to access this page.", icon="🔒")
        st.stop()


require_pin()

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class DeliveryRow:
    type_label: str  # "1 BP" / "2 BP" / "4 BP" or blank
    patient: str
    address: str
    note: str


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, COALESCE(address,'') AS address, weekday, interval_weeks, packs_per_delivery, anchor_date,
                   COALESCE(notes,'') AS notes, active
            FROM bp_patients
            WHERE active = 1
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
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
            SELECT id, odate, patient_name, action, packs, COALESCE(note,'') AS note
            FROM bp_overrides
            WHERE odate >= ? AND odate <= ?
            """,
            c,
            params=(month_start.isoformat(), month_end.isoformat()),
        )
    if df.empty:
        return df
    df["odate"] = pd.to_datetime(df["odate"], errors="coerce").dt.date
    return df


def normalize_anchor_to_weekday(anchor: date, weekday: int) -> date:
    if anchor.weekday() == weekday:
        return anchor
    delta = (weekday - anchor.weekday()) % 7
    return anchor + timedelta(days=delta)


def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    weeks_between = (d - anchor).days // 7
    return (weeks_between % interval_weeks) == 0


def build_day_rows(d: date) -> list[DeliveryRow]:
    pts = read_patients()
    if pts.empty:
        return []

    # weekday only auto-schedule
    if d.weekday() > 4:
        base_rows = []
    else:
        base_rows = []
        todays = pts[pts["weekday"] == d.weekday()]
        for _, r in todays.iterrows():
            raw_anchor = r["anchor_date"]
            if not isinstance(raw_anchor, date):
                continue
            anchor = normalize_anchor_to_weekday(raw_anchor, int(r["weekday"]))
            interval = int(r["interval_weeks"])
            if occurs_on_day(anchor, interval, d):
                packs = int(r["packs_per_delivery"])
                label = f"{packs} BP"
                note = str(r.get("notes", "") or "").strip()
                base_rows.append(
                    DeliveryRow(
                        type_label=label,
                        patient=str(r["name"]),
                        address=str(r.get("address", "") or ""),
                        note=note,
                    )
                )

    # apply overrides for month
    ms, me = month_bounds(d.year, d.month)
    ov = read_overrides(ms, me)

    if not ov.empty:
        # skip/add
        for _, r in ov[ov["odate"] == d].iterrows():
            action = str(r["action"]).strip().lower()
            pname = str(r["patient_name"]).strip()
            packs = r["packs"]
            onote = str(r.get("note", "") or "").strip()

            if action == "skip":
                base_rows = [x for x in base_rows if x.patient != pname]
            elif action == "add":
                p = int(packs) if pd.notna(packs) else 1
                base_rows.append(DeliveryRow(type_label=f"{p} BP", patient=pname, address="", note=onote))

    # sort by BP type then name
    def sort_key(x: DeliveryRow):
        # "1 BP" -> 1 ; blank -> 99
        try:
            n = int(x.type_label.split()[0])
        except Exception:
            n = 99
        return (n, x.patient.lower())

    base_rows.sort(key=sort_key)
    return base_rows


def truncate(text: str, max_w: float, font: str, size: int) -> str:
    if pdfmetrics.stringWidth(text, font, size) <= max_w:
        return text
    ell = "…"
    avail = max_w - pdfmetrics.stringWidth(ell, font, size)
    if avail <= 0:
        return ell
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if pdfmetrics.stringWidth(text[:mid], font, size) <= avail:
            lo = mid + 1
        else:
            hi = mid
    cut = max(0, lo - 1)
    return text[:cut].rstrip() + ell


def make_daily_pdf(
    d: date,
    rows: list[DeliveryRow],
    extra_blank_rows: int,
    paper: str = "letter",
    orientation: str = "portrait",
) -> bytes:
    base = letter if paper == "letter" else legal
    pagesize = landscape(base) if orientation == "landscape" else portrait(base)

    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_daily.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    # margins
    left = 0.50 * inch
    right = w - 0.50 * inch
    top = h - 0.50 * inch
    bottom = 0.50 * inch

    # header center
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, top, "SDM DELIVERY SHEET LOG")

    c.setFont("Helvetica-Bold", 12)
    title2 = f"{d.strftime('%A')} — {d.isoformat()}"
    c.drawCentredString(w / 2, top - 0.28 * inch, title2)

    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.28 * inch, f"Generated: {now_local().strftime('%Y-%m-%d %H:%M')}  ({TZ_NAME})")

    # table geometry
    table_top = top - 0.55 * inch
    table_bottom = bottom
    table_h = table_top - table_bottom

    # columns: Type | Patient | Address | Notes | Packages | Charged
    col_w = [
        0.95 * inch,   # Type
        2.15 * inch,   # Patient
        3.85 * inch,   # Address (big)
        1.25 * inch,   # Notes (smaller)
        0.85 * inch,   # Packages
        0.85 * inch,   # Charged
    ]
    total_w = sum(col_w)
    scale = (right - left) / total_w
    col_w = [x * scale for x in col_w]

    headers = ["Type", "Patient", "Address", "Notes", "Packages", "Charged"]

    # rows count
    total_rows = len(rows) + int(extra_blank_rows)
    if total_rows < 12:
        total_rows = 12  # keep some writing room

    row_h = table_h / (total_rows + 1)  # +1 for header row

    # header background
    c.setFillGray(0.92)
    c.rect(left, table_top - row_h, right - left, row_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # draw header text
    c.setFont("Helvetica-Bold", 10)
    x = left
    for i, htxt in enumerate(headers):
        c.drawString(x + 4, table_top - row_h + 6, htxt)
        x += col_w[i]

    # grid lines
    c.setStrokeGray(0.65)
    c.rect(left, table_bottom, right - left, table_top - table_bottom, stroke=1, fill=0)

    # vertical lines
    x = left
    for cw in col_w[:-1]:
        x += cw
        c.line(x, table_bottom, x, table_top)

    # horizontal lines
    y = table_top
    for _ in range(total_rows + 1):
        c.line(left, y, right, y)
        y -= row_h

    # fill rows
    c.setFont("Helvetica", 9)
    y = table_top - row_h
    font = "Helvetica"
    size = 9

    def cell_text(col_index: int, text: str) -> str:
        max_w = col_w[col_index] - 8
        return truncate(text, max_w, font, size)

    for r in rows:
        y -= row_h
        if y < table_bottom:
            break
        c.drawString(left + 4, y + 6, cell_text(0, r.type_label))
        c.drawString(left + col_w[0] + 4, y + 6, cell_text(1, r.patient))
        c.drawString(left + col_w[0] + col_w[1] + 4, y + 6, cell_text(2, r.address))
        c.drawString(left + col_w[0] + col_w[1] + col_w[2] + 4, y + 6, cell_text(3, r.note))
        # Packages + Charged left EMPTY (manual fill)

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

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    paper = st.selectbox("Paper size", ["letter", "legal"], index=0)
with c2:
    orientation = st.selectbox("Print layout", ["portrait", "landscape"], index=0)
with c3:
    extra = st.slider("Extra blank lines (for extra deliveries)", 0, 60, 20)

rows = build_day_rows(pick)

st.subheader("Auto-filled BP deliveries for this day")
if not rows:
    st.info("No BP deliveries matched for this day (or it’s weekend). Overrides can still add deliveries.")
else:
    df = pd.DataFrame([{"Type": r.type_label, "Patient": r.patient, "Address": r.address, "Notes": r.note} for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)

pdf = make_daily_pdf(
    pick,
    rows,
    extra_blank_rows=extra,
    paper=paper,
    orientation=orientation,
)

st.download_button(
    "Download Daily PDF (SDM DELIVERY SHEET LOG)",
    data=pdf,
    file_name=f"sdm_delivery_sheet_{pick.isoformat()}_{orientation}.pdf",
    mime="application/pdf",
    type="primary",
)
