# pages/5_Blisterpack_Tracker.py
# Blister Pack Delivery Sheet (Auto Month Generator) + Daily "SDM DELIVERY SHEET LOG"
# ✅ Keeps SAME SQLite DB (data/blisterpacks.db) so your data won't vanish
# ✅ Calendar view default = Biweekly+Monthly (tab order)
# ✅ Month PDF downloads: Weekly / Biweekly+Monthly / All (ONE PAGE, Landscape)
# ✅ Daily Delivery Sheet PDF: auto-fills BP patients for the day + extra blank RX lines
# ✅ Filters + Quick Add + Delete + CSV Import/Export (no Excel dependency)

import os
import io
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
from reportlab.pdfbase import pdfmetrics


# =========================
# Page config + PIN lock
# =========================
st.set_page_config(page_title="Blister Pack Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))
APP_TZ = str(st.secrets.get("APP_TZ", "America/Toronto"))  # change in secrets if needed

def now_local() -> datetime:
    return datetime.now(ZoneInfo(APP_TZ))

def stamp() -> str:
    # includes TZ abbreviation so you can verify it's correct
    return now_local().strftime("%Y-%m-%d %H:%M %Z")

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
    st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


# =========================
# Header
# =========================
st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption(
    "Auto-generates your month delivery sheet from patient frequency: Weekly / Biweekly / Monthly (4-week). "
    "Use Overrides for holidays/exceptions."
)

# =========================
# SQLite setup (KEEP SAME PATH)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)

def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _col_exists(c: sqlite3.Connection, table: str, col: str) -> bool:
    cur = c.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols

def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT,                            -- added later (migration-safe)
                weekday INTEGER NOT NULL,                 -- 0=Mon ... 6=Sun
                interval_weeks INTEGER NOT NULL,          -- 1 / 2 / 4
                packs_per_delivery INTEGER NOT NULL,      -- usually matches interval (1/2/4)
                anchor_date TEXT NOT NULL,                -- ISO yyyy-mm-dd
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        # Migration: if older DB existed without address, add it
        if not _col_exists(c, "bp_patients", "address"):
            try:
                c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT")
            except Exception:
                pass

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,                      -- ISO yyyy-mm-dd
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,                     -- 'skip' or 'add'
                packs INTEGER,                            -- used for 'add'
                note TEXT
            )
            """
        )

        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_patients_weekday ON bp_patients(weekday)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_overrides_date ON bp_overrides(odate)")
        c.commit()

init_db()

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SUN_FIRST = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
FREQ_LABEL = {1: "Weekly", 2: "Biweekly", 4: "Monthly"}
LABEL_TO_FREQ = {"Weekly": 1, "Biweekly": 2, "Monthly": 4}


# =========================
# DB functions
# =========================
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
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["active"] = df["active"].astype(bool)
    df["address"] = df["address"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df

def insert_patient(
    name: str,
    address: str,
    weekday: int,
    interval_weeks: int,
    packs_per_delivery: int,
    anchor_date: date,
    notes: str,
    active: bool,
):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_patients (name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                (address or "").strip(),
                int(weekday),
                int(interval_weeks),
                int(packs_per_delivery),
                anchor_date.isoformat(),
                (notes or "").strip(),
                1 if active else 0,
            ),
        )
        c.commit()

def upsert_patients(df: pd.DataFrame):
    # df may come from editor/import
    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = str(r.get("name", "")).strip()
            if not name:
                continue

            address = "" if pd.isna(r.get("address", "")) else str(r.get("address", "")).strip()
            weekday = int(r.get("weekday", 0))
            interval = int(r.get("interval_weeks", 1))
            packs = int(r.get("packs_per_delivery", interval))
            anchor = r.get("anchor_date", date.today())

            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor, errors="coerce").date()

            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))
            active = 1 if bool(r.get("active", True)) else 0

            if pd.isna(rid) or rid is None:
                c.execute(
                    """
                    INSERT INTO bp_patients (name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, address, weekday, interval, packs, anchor.isoformat(), notes, active),
                )
            else:
                c.execute(
                    """
                    UPDATE bp_patients
                    SET name=?, address=?, weekday=?, interval_weeks=?, packs_per_delivery=?, anchor_date=?, notes=?, active=?
                    WHERE id=?
                    """,
                    (name, address, weekday, interval, packs, anchor.isoformat(), notes, active, int(rid)),
                )
        c.commit()

def delete_patient_by_id(pid: int):
    with conn() as c:
        c.execute("DELETE FROM bp_patients WHERE id=?", (pid,))
        c.commit()

def wipe_patients():
    with conn() as c:
        c.execute("DELETE FROM bp_patients")
        c.commit()

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
            ORDER BY odate ASC, patient_name ASC
            """,
            c,
            params=(month_start.isoformat(), month_end.isoformat()),
        )
    if df.empty:
        return df
    df["odate"] = pd.to_datetime(df["odate"], errors="coerce").dt.date
    df["note"] = df["note"].fillna("")
    return df

def add_override(odate: date, patient_name: str, action: str, packs: int | None, note: str):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_overrides (odate, patient_name, action, packs, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (odate.isoformat(), patient_name.strip(), action.strip(), packs, (note or "").strip()),
        )
        c.commit()

def delete_override(oid: int):
    with conn() as c:
        c.execute("DELETE FROM bp_overrides WHERE id=?", (oid,))
        c.commit()


# =========================
# Scheduling logic
# =========================
def dates_in_month(year: int, month: int) -> list[date]:
    start, end = month_bounds(year, month)
    d = start
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out

def occurs_on_day(anchor: date, interval_weeks: int, d: date) -> bool:
    if d < anchor:
        return False
    delta_days = (d - anchor).days
    weeks_between = delta_days // 7
    return (weeks_between % interval_weeks) == 0

@dataclass
class DeliveryItem:
    pid: int | None
    name: str
    packs: int
    interval_weeks: int  # 1 weekly, 2 biweekly, 4 monthly, 99 manual add
    address: str = ""
    notes: str = ""

def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # Automatic schedule: weekdays only
        if d.weekday() > 4:
            continue

        todays = active[active["weekday"] == d.weekday()]
        if todays.empty:
            continue

        for _, r in todays.iterrows():
            anchor = r["anchor_date"]
            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            interval = int(r["interval_weeks"])

            if occurs_on_day(anchor, interval, d):
                schedule[d].append(
                    DeliveryItem(
                        pid=None if pd.isna(r.get("id", None)) else int(r["id"]),
                        name=str(r["name"]),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=interval,
                        address=str(r.get("address", "") or ""),
                        notes=str(r.get("notes", "") or ""),
                    )
                )

        # weekly -> biweekly -> monthly, then name
        schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return schedule

def apply_overrides(schedule: dict[date, list[DeliveryItem]], overrides_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    if overrides_df.empty:
        return schedule

    for _, r in overrides_df.iterrows():
        d = r["odate"]
        name = str(r["patient_name"])
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])
        note = str(r.get("note", "") or "")

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x.name != name]
        elif action == "add":
            schedule[d].append(
                DeliveryItem(
                    pid=None,
                    name=name,
                    packs=packs or 1,
                    interval_weeks=99,   # manual = after monthly
                    address="",
                    notes=note,
                )
            )
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return schedule

def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
    """
    mode:
      - "Weekly" -> {1} plus manual 99
      - "Biweekly + Monthly" -> {2,4} plus manual 99
      - "All" -> {1,2,4} plus manual 99
    """
    if mode == "Weekly":
        allowed = {1, 99}
    elif mode == "Biweekly + Monthly":
        allowed = {2, 4, 99}
    else:
        allowed = {1, 2, 4, 99}

    out: dict[date, list[DeliveryItem]] = {}
    for d, items in schedule.items():
        out[d] = [x for x in items if x.interval_weeks in allowed]
        out[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return out


# =========================
# PDF helpers (Month)
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

def make_month_pdf_one_page(
    year: int,
    month: int,
    schedule: dict[date, list[DeliveryItem]],
    page_mode: str = "letter",        # "letter" or "legal"
    min_font: int = 3,
    allow_two_columns: bool = True,
) -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_month_onepage.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.20 * inch
    left = margin
    right = w - margin
    bottom = margin
    top = h - margin

    # compact header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top - 0.12 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.30 * inch, f"Generated: {stamp()}")

    grid_top = top - 0.45 * inch
    grid_bottom = bottom
    grid_left = left
    grid_right = right

    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    col_w = grid_w / 7.0

    header_h = 0.22 * inch
    body_h = grid_h - header_h

    cal = calendar.Calendar(firstweekday=6)  # Sunday first
    weeks = cal.monthdatescalendar(year, month)
    rows = len(weeks)
    row_h = body_h / rows

    # header background
    c.setFillGray(0.92)
    c.rect(grid_left, grid_top - header_h, grid_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 9)
    for i, lbl in enumerate(SUN_FIRST):
        c.drawString(grid_left + i * col_w + 3, grid_top - header_h + 6, lbl)

    # outer border
    c.setStrokeGray(0.65)
    c.rect(grid_left, grid_bottom, grid_w, grid_h, stroke=1, fill=0)
    c.line(grid_left, grid_top - header_h, grid_right, grid_top - header_h)

    font_name = "Helvetica"

    for r, week in enumerate(weeks):
        y_top = grid_top - header_h - r * row_h
        y_bot = y_top - row_h

        c.setStrokeGray(0.65)
        c.line(grid_left, y_top, grid_right, y_top)

        for col, d in enumerate(week):
            x0 = grid_left + col * col_w

            c.setStrokeGray(0.65)
            c.line(x0, y_bot, x0, y_top)

            # Only selected month: other-month cells blank
            if d.month != month:
                continue

            c.setFont("Helvetica-Bold", 9)
            c.drawString(x0 + 3, y_top - 12, str(d.day))

            items = schedule.get(d, [])
            if not items:
                continue

            # cell text area
            pad_x = 3
            pad_y_top = 18
            pad_y_bottom = 3
            area_top = y_top - pad_y_top
            area_bottom = y_bot + pad_y_bottom
            area_h = max(0, area_top - area_bottom)
            if area_h <= 0:
                continue

            # NO weird W/B/M prefixes – just name + packs
            entries = [f"{it.name} ({it.packs}p)" for it in items]

            def fits_single(fs: int) -> tuple[bool, int]:
                line_h = fs + 0.5
                max_lines = int(area_h // line_h)
                return (len(entries) <= max_lines, max_lines)

            def fits_two(fs: int) -> tuple[bool, int]:
                if not allow_two_columns:
                    return (False, 0)
                line_h = fs + 0.5
                max_lines = int(area_h // line_h)
                return (len(entries) <= max_lines * 2, max_lines)

            chosen_mode = None
            chosen_fs = None
            chosen_max_lines = None

            for fs in [8, 7, 6, 5, 4, 3]:
                if fs < min_font:
                    continue
                ok, max_lines = fits_single(fs)
                if ok and max_lines > 0:
                    chosen_mode, chosen_fs, chosen_max_lines = "single", fs, max_lines
                    break

            if chosen_mode is None:
                for fs in [7, 6, 5, 4, 3]:
                    if fs < min_font:
                        continue
                    ok, max_lines = fits_two(fs)
                    if ok and max_lines > 0:
                        chosen_mode, chosen_fs, chosen_max_lines = "two", fs, max_lines
                        break

            if chosen_mode is None:
                chosen_mode = "two" if allow_two_columns else "single"
                chosen_fs = min_font
                chosen_max_lines = int(area_h // (chosen_fs + 0.5))
                if chosen_max_lines < 1:
                    continue

            fs = chosen_fs
            c.setFont(font_name, fs)
            line_h = fs + 0.5

            if chosen_mode == "single":
                max_w = col_w - 2 * pad_x
                cap = chosen_max_lines
                to_print = entries[:cap]
                remaining = len(entries) - len(to_print)
                if remaining > 0 and cap >= 1:
                    to_print = entries[:cap - 1]
                    to_print.append(f"+{remaining} more")
                y = area_top - fs
                for i, e in enumerate(to_print):
                    c.drawString(x0 + pad_x, y - i * line_h, truncate_to_width(e, max_w, font_name, fs))
            else:
                gap = 6
                half_w = (col_w - 2 * pad_x - gap) / 2.0
                x_left = x0 + pad_x
                x_right = x0 + pad_x + half_w + gap

                cap = chosen_max_lines * 2
                to_print = entries[:cap]
                remaining = len(entries) - len(to_print)
                if remaining > 0 and cap >= 1:
                    to_print = entries[:cap - 1]
                    to_print.append(f"+{remaining} more")

                y = area_top - fs
                for i, e in enumerate(to_print):
                    line = truncate_to_width(e, half_w, font_name, fs)
                    if i < chosen_max_lines:
                        c.drawString(x_left, y - i * line_h, line)
                    else:
                        j = i - chosen_max_lines
                        c.drawString(x_right, y - j * line_h, line)

    c.setStrokeGray(0.65)
    c.line(grid_right, grid_bottom, grid_right, grid_top)
    c.line(grid_left, grid_bottom, grid_right, grid_bottom)

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
# PDF helpers (Weekly optional)
# =========================
def wrap_text_to_width(text: str, max_width: float, font_name: str, font_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for w in words[1:]:
        cand = cur + " " + w
        if pdfmetrics.stringWidth(cand, font_name, font_size) <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def make_week_pdf(week_start: date, schedule: dict[date, list[DeliveryItem]], page_mode: str = "letter") -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_week.pdf")

    c = canvas.Canvas(tmp_path, pagesize=pagesize)
    week_end = week_start + timedelta(days=6)

    title = f"Blister Pack — Weekly Delivery Sheet ({week_start.isoformat()} to {week_end.isoformat()})"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5 * inch, h - 0.55 * inch, title)

    c.setFont("Helvetica", 9)
    c.drawString(0.5 * inch, h - 0.8 * inch, f"Generated: {stamp()}")

    left = 0.5 * inch
    right = w - 0.5 * inch
    top = h - 1.05 * inch
    bottom = 0.5 * inch

    col_day = 2.2 * inch
    row_h = (top - bottom) / 8.0

    c.setFillGray(0.92)
    c.rect(left, top - row_h, right - left, row_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left + 6, top - row_h + 8, "Day")
    c.drawString(left + col_day + 6, top - row_h + 8, "Due (Weekly → Biweekly → Monthly)   [ ] Delivered   Notes")

    y = top - row_h
    for i in range(7):
        d = week_start + timedelta(days=i)
        y2 = y - row_h

        c.setStrokeGray(0.75)
        c.line(left, y, right, y)
        c.line(left, y2, right, y2)
        c.line(left, y2, left, y)
        c.line(left + col_day, y2, left + col_day, y)
        c.line(right, y2, right, y)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(left + 6, y2 + row_h - 14, f"{WEEKDAY_LABELS[d.weekday()]} {d.isoformat()}")

        items = schedule.get(d, [])
        items = sorted(items, key=lambda x: (x.interval_weeks, x.name.lower()))

        c.setFont("Helvetica", 9)
        yy = y2 + row_h - 30
        line_h = 11
        max_w = (right - left - col_day - 16)

        for it in items:
            line = f"{it.name} ({it.packs}p)"
            for wline in wrap_text_to_width(line, max_w, "Helvetica", 9):
                c.drawString(left + col_day + 10, yy, wline)
                yy -= line_h
                if yy < y2 + 6:
                    break
            if yy < y2 + 6:
                break

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
# PDF helpers (Daily SDM Delivery Sheet Log)
# =========================
def make_daily_delivery_pdf(
    d: date,
    items: list[DeliveryItem],
    extra_rx_lines: int = 12,
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

    # Header (center)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2, top - 0.10 * inch, "SDM DELIVERY SHEET LOG")

    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, top - 0.34 * inch, f"{d.strftime('%A')} — {d.isoformat()}")

    c.setFont("Helvetica", 9)
    c.drawRightString(right, top - 0.34 * inch, f"Generated: {stamp()}")

    # Table setup
    table_top = top - 0.55 * inch
    table_bottom = bottom
    table_h = table_top - table_bottom

    header_h = 0.28 * inch
    row_h = 0.30 * inch  # writing space

    # Column widths (sum must fit)
    # Type (short), Patient, Address (big), Notes (smaller), Packages, Charged
    col_w = {
        "type": 0.85 * inch,
        "patient": 2.35 * inch,
        "address": 3.65 * inch,
        "notes": 1.35 * inch,      # smaller as requested
        "packages": 0.95 * inch,
        "charged": 0.90 * inch,
    }
    total_w = sum(col_w.values())
    avail_w = (right - left)
    if total_w > avail_w:
        # shrink address first if printer is tight
        shrink = total_w - avail_w
        col_w["address"] = max(2.8 * inch, col_w["address"] - shrink)

    # rows that fit
    max_rows = int((table_h - header_h) // row_h)
    if max_rows < 5:
        max_rows = 5

    # Prepare rows: BP scheduled + RX blanks (clipped to fit 1 page)
    rows = []

    # Items already ordered by interval in schedule; keep that
    for it in items:
        # Type should look like: "1 BP", "2 BP", "4 BP", or "BP" for manual adds
        if it.interval_weeks in (1, 2, 4):
            t = f"{it.packs} BP"
        else:
            t = "BP"
        rows.append([t, it.name, it.address or "", "", "", ""])  # notes, packages, charged empty

    # Add RX blank lines
    for _ in range(extra_rx_lines):
        rows.append(["RX", "", "", "", "", ""])

    # Clip rows to one page
    rows = rows[:max_rows]

    # Header row background
    c.setFillGray(0.92)
    c.rect(left, table_top - header_h, avail_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)

    # Draw header text
    c.setFont("Helvetica-Bold", 10)
    x = left
    y_header = table_top - header_h + 0.09 * inch
    headers = [("type", "Type"), ("patient", "Patient"), ("address", "Address"), ("notes", "Notes"),
               ("packages", "Packages"), ("charged", "Charged")]
    for key, label in headers:
        c.drawString(x + 4, y_header, label)
        x += col_w[key]

    # Grid lines + content
    c.setStrokeGray(0.70)

    # Vertical lines
    x = left
    c.line(left, table_bottom, left, table_top)
    for key, _ in headers:
        x += col_w[key]
        c.line(x, table_bottom, x, table_top)

    # Horizontal lines (header + rows)
    c.line(left, table_top, left + avail_w, table_top)
    c.line(left, table_top - header_h, left + avail_w, table_top - header_h)

    # Content font
    c.setFont("Helvetica", 9)

    y = table_top - header_h
    for r in range(max_rows):
        y2 = y - row_h
        c.line(left, y2, left + avail_w, y2)

        if r < len(rows):
            row = rows[r]
            # Type, Patient, Address (truncate), Notes(empty), Packages(empty), Charged(empty)
            x = left
            # Type
            c.drawString(x + 4, y2 + 0.10 * inch, row[0])
            x += col_w["type"]

            # Patient
            c.drawString(x + 4, y2 + 0.10 * inch, truncate_to_width(row[1], col_w["patient"] - 8, "Helvetica", 9))
            x += col_w["patient"]

            # Address
            c.drawString(x + 4, y2 + 0.10 * inch, truncate_to_width(row[2], col_w["address"] - 8, "Helvetica", 9))
            x += col_w["address"]

            # Notes (blank but space exists)
            x += col_w["notes"]

            # Packages blank
            x += col_w["packages"]

            # Charged blank
            x += col_w["charged"]

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
# CSV Import / Export helpers
# =========================
EXPORT_COLS = ["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]

def patients_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df2 = df.copy()
    if "anchor_date" in df2.columns:
        df2["anchor_date"] = df2["anchor_date"].astype(str)
    df2.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")

def normalize_import_df(df: pd.DataFrame) -> pd.DataFrame:
    # Accept exports or user CSV with these columns
    # Required: name, weekday, interval_weeks, packs_per_delivery, anchor_date, active
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # allow some common variants
    rename_map = {
        "packs": "packs_per_delivery",
        "packs_per_deliver": "packs_per_delivery",
        "anchor": "anchor_date",
        "frequency": "interval_weeks",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    if "address" not in df.columns:
        df["address"] = ""

    if "notes" not in df.columns:
        df["notes"] = ""

    # active normalization
    if "active" not in df.columns:
        df["active"] = True
    else:
        df["active"] = df["active"].astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
        ).fillna(True)

    # enforce types
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").fillna(0).astype(int)
    df["interval_weeks"] = pd.to_numeric(df["interval_weeks"], errors="coerce").fillna(1).astype(int)
    df["packs_per_delivery"] = pd.to_numeric(df["packs_per_delivery"], errors="coerce").fillna(df["interval_weeks"]).astype(int)

    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df = df[df["name"].astype(str).str.strip() != ""]
    return df[["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]].copy()


# =========================
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs"]
)

today = date.today()


# -------------------------
# Patients tab (filters + quick add + delete + import/export)
# -------------------------
with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption("Frequency controls automation. Anchor date defines the cycle start. Address is used in the Daily sheet PDF.")

    # Quick add (so you don't fight the editor)
    with st.expander("➕ Quick Add Patient", expanded=True):
        a1, a2, a3, a4 = st.columns([2.0, 1.2, 1.2, 1.4])
        with a1:
            new_name = st.text_input("Patient name", key="qa_name")
            new_address = st.text_input("Address (optional)", key="qa_addr")
        with a2:
            new_weekday = st.selectbox("Weekday (Mon–Fri)", [0, 1, 2, 3, 4], format_func=lambda x: WEEKDAY_LABELS[x], key="qa_wd")
            new_freq_label = st.selectbox("Frequency", ["Weekly", "Biweekly", "Monthly"], key="qa_freq")
        with a3:
            new_anchor = st.date_input("Anchor date", value=today, key="qa_anchor")
            new_active = st.checkbox("Active", value=True, key="qa_active")
        with a4:
            new_notes = st.text_input("Notes (optional)", key="qa_notes")
            auto_fix_packs = st.checkbox("Auto-fix packs = frequency", value=True, key="qa_autofix")
            packs_val = LABEL_TO_FREQ[new_freq_label] if auto_fix_packs else st.selectbox("Packs per delivery", [1, 2, 4], key="qa_packs")

        if st.button("Add patient", type="primary"):
            if not new_name.strip():
                st.error("Name is required.")
            else:
                interval = LABEL_TO_FREQ[new_freq_label]
                packs = interval if auto_fix_packs else int(packs_val)
                insert_patient(
                    name=new_name,
                    address=new_address,
                    weekday=int(new_weekday),
                    interval_weeks=int(interval),
                    packs_per_delivery=int(packs),
                    anchor_date=new_anchor,
                    notes=new_notes,
                    active=bool(new_active),
                )
                st.success("Patient added.")
                st.rerun()

    df_all = read_patients()
    if df_all.empty:
        df_all = pd.DataFrame(columns=EXPORT_COLS)

    st.divider()
    st.markdown("### Filters (for easier navigation)")
    f1, f2, f3, f4 = st.columns([1.6, 1.1, 1.2, 1.0])
    with f1:
        q = st.text_input("Search name / address / notes", value="")
    with f2:
        freq_pick = st.multiselect(
            "Frequency",
            options=["Weekly", "Biweekly", "Monthly"],
            default=["Weekly", "Biweekly", "Monthly"],
        )
    with f3:
        wd_pick = st.multiselect(
            "Weekday",
            options=["Mon", "Tue", "Wed", "Thu", "Fri"],
            default=["Mon", "Tue", "Wed", "Thu", "Fri"],
        )
    with f4:
        active_only = st.toggle("Active only", value=False)

    df_view = df_all.copy()

    if q.strip():
        qq = q.strip().lower()
        df_view = df_view[
            df_view["name"].astype(str).str.lower().str.contains(qq)
            | df_view["address"].astype(str).str.lower().str.contains(qq)
            | df_view["notes"].astype(str).str.lower().str.contains(qq)
        ]

    if freq_pick:
        allowed = {LABEL_TO_FREQ[x] for x in freq_pick}
        df_view = df_view[df_view["interval_weeks"].isin(list(allowed))]

    if wd_pick:
        allowed_wd = {WEEKDAY_LABELS.index(x) for x in wd_pick}
        df_view = df_view[df_view["weekday"].isin(list(allowed_wd))]

    if active_only and not df_view.empty:
        df_view = df_view[df_view["active"] == True]

    if "__delete__" not in df_view.columns:
        df_view["__delete__"] = False

    st.markdown("### Edit existing patients (checkbox Delete if needed)")
    edited = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="fixed",  # less buggy for adding; use Quick Add above instead
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Patient name (printed)", required=True),
            "address": st.column_config.TextColumn("Address (daily sheet)", required=False),
            "weekday": st.column_config.SelectboxColumn(
                "Delivery weekday (Mon–Fri)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: WEEKDAY_LABELS[int(x)],
                required=True,
            ),
            "interval_weeks": st.column_config.SelectboxColumn(
                "Frequency",
                options=[1, 2, 4],
                format_func=lambda v: FREQ_LABEL.get(int(v), str(v)),
                required=True,
            ),
            "packs_per_delivery": st.column_config.SelectboxColumn("Packs per delivery", options=[1, 2, 4], required=True),
            "anchor_date": st.column_config.DateColumn("Anchor date", required=True),
            "notes": st.column_config.TextColumn("Notes"),
            "active": st.column_config.CheckboxColumn("Active"),
            "__delete__": st.column_config.CheckboxColumn("Delete"),
        },
        hide_index=True,
    )

    auto_fix = st.toggle("Auto-fix packs to match frequency (recommended)", value=True)
    if auto_fix and not edited.empty:
        for idx in edited.index:
            try:
                interval = int(edited.loc[idx, "interval_weeks"])
                edited.loc[idx, "packs_per_delivery"] = interval
            except Exception:
                pass

    if st.button("Save changes", type="primary"):
        if edited.empty:
            st.info("Nothing to save.")
            st.stop()

        # Delete marked
        to_delete = edited[(edited["__delete__"] == True) & (~edited["id"].isna())]
        for _, r in to_delete.iterrows():
            delete_patient_by_id(int(r["id"]))

        # Upsert others
        keep = edited[edited["__delete__"] == False].drop(columns=["__delete__"], errors="ignore")

        bad = keep[keep["name"].astype(str).str.strip() == ""]
        if not bad.empty:
            st.error("Some rows have empty names. Fix them or delete those rows.")
            st.stop()

        upsert_patients(keep)
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.subheader("Import / Export (restore missing data)")

    # Export buttons
    csv_bytes = patients_to_csv_bytes(read_patients())
    st.download_button(
        "⬇️ Download patients CSV (backup)",
        data=csv_bytes,
        file_name=f"bp_patients_export_{now_local().strftime('%Y-%m-%d_%H%M')}.csv",
        mime="text/csv",
    )
    try:
        with open(DB_PATH, "rb") as f:
            db_bytes = f.read()
        st.download_button(
            "⬇️ Download SQLite DB (full backup)",
            data=db_bytes,
            file_name="blisterpacks.db",
            mime="application/octet-stream",
        )
    except Exception:
        st.info("DB file not readable for download right now (permissions).")

    st.markdown("### Import from CSV (recommended)")
    up = st.file_uploader("Upload your exported CSV", type=["csv"])
    replace = st.toggle("Replace existing patients (wipe then restore)", value=False)
    if up is not None:
        try:
            imp = pd.read_csv(up)
            imp = normalize_import_df(imp)
            st.write("Preview:")
            st.dataframe(imp.head(25), use_container_width=True, hide_index=True)

            if st.button("Import CSV now", type="primary"):
                if replace:
                    wipe_patients()
                # keep ids if present? we will ignore ids on insert if replace is on; but upsert can keep ids too
                upsert_patients(imp)
                st.success("Import complete.")
                st.rerun()
        except Exception as e:
            st.error(f"Import failed. Make sure it's a CSV export. Error: {e}")


# -------------------------
# Overrides tab
# -------------------------
with tab_overrides:
    st.subheader("Overrides (manual exceptions)")
    st.caption("Use overrides for holidays/patient-specific changes: skip removes, add inserts extra delivery (even weekends).")

    o1, o2 = st.columns([1, 1])
    with o1:
        oy = st.number_input("Override year", 2020, 2100, today.year, 1, key="oy")
    with o2:
        om = st.selectbox(
            "Override month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: calendar.month_name[m],
            key="om",
        )

    mstart, mend = month_bounds(int(oy), int(om))
    patients_df = read_patients()
    names = sorted(patients_df["name"].tolist()) if not patients_df.empty else []

    st.markdown("### Add override")
    oc1, oc2, oc3, oc4, oc5 = st.columns([1.1, 1.6, 1.2, 0.9, 1.4])
    with oc1:
        odate = st.date_input("Date", value=mstart, min_value=mstart, max_value=mend)
    with oc2:
        pname = st.selectbox("Patient", names) if names else st.text_input("Patient name")
    with oc3:
        action = st.selectbox("Action", ["skip", "add"])
    with oc4:
        packs = None
        if action == "add":
            packs = st.number_input("Packs", min_value=1, max_value=10, value=1, step=1)
        else:
            st.write("")
    with oc5:
        note = st.text_input("Note", placeholder="e.g., holiday change / patient ok pickup")

    if st.button("Save override", type="primary"):
        if not pname or str(pname).strip() == "":
            st.error("Pick a patient.")
        else:
            add_override(odate, str(pname), action, int(packs) if packs is not None else None, note)
            st.success("Override saved.")
            st.rerun()

    st.divider()
    st.markdown("### Existing overrides (this month)")
    ov = read_overrides(mstart, mend)
    if ov.empty:
        st.info("No overrides yet.")
    else:
        st.dataframe(ov, use_container_width=True, hide_index=True)
        del_id = st.number_input("Override ID to delete", min_value=1, step=1)
        confirm = st.checkbox("Confirm delete override")
        if st.button("Delete override", disabled=not confirm):
            delete_override(int(del_id))
            st.success("Deleted override.")
            st.rerun()


# -------------------------
# Calendar tab (DEFAULT = Biweekly+Monthly)
# -------------------------
with tab_cal:
    c1, c2 = st.columns([1, 1])
    with c1:
        year = st.number_input("Year", min_value=2020, max_value=2100, value=today.year, step=1)
    with c2:
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: calendar.month_name[m],
        )

    patients_df = read_patients()
    base = build_month_schedule(int(year), int(month), patients_df)
    start, end = month_bounds(int(year), int(month))
    overrides_df = read_overrides(start, end)
    schedule_all = apply_overrides(base, overrides_df)

    # tabs like your screenshot (default first)
    t_bm, t_w, t_all = st.tabs(["Biweekly + Monthly", "Weekly", "All"])

    st.markdown(
        """
        <style>
        .bp-cell { border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 8px; min-height: 140px; }
        .bp-date { font-weight: 800; font-size: 14px; margin-bottom: 6px; }
        .bp-muted { opacity: 0.20; }
        .bp-item { font-size: 12px; line-height: 1.22; margin: 0 0 2px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .bp-more { font-size: 12px; opacity: 0.7; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(int(year), int(month))

    def render_calendar(mode: str):
        schedule = filter_schedule(schedule_all, mode)

        st.subheader(f"{calendar.month_name[int(month)]} {int(year)} — {mode}")
        hdr = st.columns(7)
        for i, lbl in enumerate(SUN_FIRST):
            hdr[i].markdown(f"**{lbl}**")

        for week in weeks:
            cols = st.columns(7)
            for i, d in enumerate(week):
                if d.month != int(month):
                    cols[i].markdown("<div class='bp-cell bp-muted'></div>", unsafe_allow_html=True)
                    continue

                items = schedule.get(d, [])
                items = sorted(items, key=lambda x: (x.interval_weeks, x.name.lower()))
                shown = items[:14]
                extra = max(0, len(items) - len(shown))

                lines = []
                for it in shown:
                    lines.append(f"<div class='bp-item'>{it.name} <span style='opacity:0.7'>({it.packs}p)</span></div>")

                more = f"<div class='bp-more'>+{extra} more</div>" if extra > 0 else ""

                html = f"""
                <div class="bp-cell">
                  <div class="bp-date">{d.day}</div>
                  {''.join(lines)}
                  {more}
                </div>
                """
                cols[i].markdown(html, unsafe_allow_html=True)

    with t_bm:
        render_calendar("Biweekly + Monthly")
    with t_w:
        render_calendar("Weekly")
    with t_all:
        render_calendar("All")


# -------------------------
# Print tab (Month PDFs + Weekly optional + Daily Delivery Sheet)
# -------------------------
with tab_print:
    st.subheader("Print PDFs (Landscape • One Page • Selected Month Only)")

    pc1, pc2, pc3 = st.columns([1, 1, 1.2])
    with pc1:
        py = st.number_input("Print year", 2020, 2100, today.year, 1, key="py")
    with pc2:
        pm = st.selectbox(
            "Print month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: calendar.month_name[m],
            key="pm",
        )
    with pc3:
        page_mode = st.selectbox("Paper size", ["letter", "legal"], index=0)
        st.caption("Legal gives more space if your printer supports it.")

    allow_two_cols = st.toggle("Max fit (use 2 columns inside busy days)", value=True)
    min_font = st.slider("Minimum font size (smaller = fits more)", min_value=3, max_value=6, value=3)

    patients_df = read_patients()
    base = build_month_schedule(int(py), int(pm), patients_df)
    mstart, mend = month_bounds(int(py), int(pm))
    overrides_df = read_overrides(mstart, mend)
    sched_all = apply_overrides(base, overrides_df)

    # Download buttons like you asked: Weekly AND Biweekly+Monthly together
    colA, colB, colC = st.columns(3)

    with colA:
        sched_weekly = filter_schedule(sched_all, "Weekly")
        pdf_weekly = make_month_pdf_one_page(int(py), int(pm), sched_weekly, page_mode=page_mode, min_font=int(min_font), allow_two_columns=allow_two_cols)
        st.download_button(
            "Download Month PDF — Weekly",
            data=pdf_weekly,
            file_name=f"bp_month_{py}_{pm:02d}_weekly.pdf",
            mime="application/pdf",
            type="primary",
        )

    with colB:
        sched_bm = filter_schedule(sched_all, "Biweekly + Monthly")
        pdf_bm = make_month_pdf_one_page(int(py), int(pm), sched_bm, page_mode=page_mode, min_font=int(min_font), allow_two_columns=allow_two_cols)
        st.download_button(
            "Download Month PDF — Biweekly+Monthly",
            data=pdf_bm,
            file_name=f"bp_month_{py}_{pm:02d}_biweekly_monthly.pdf",
            mime="application/pdf",
            type="primary",
        )

    with colC:
        pdf_all = make_month_pdf_one_page(int(py), int(pm), sched_all, page_mode=page_mode, min_font=int(min_font), allow_two_columns=allow_two_cols)
        st.download_button(
            "Download Month PDF — All",
            data=pdf_all,
            file_name=f"bp_month_{py}_{pm:02d}_all.pdf",
            mime="application/pdf",
        )

    st.caption("If a day has too many names, the PDF shows “+X more” instead of silently dropping patients.")

    st.divider()
    st.markdown("### Weekly PDF (optional)")
    any_day = st.date_input("Pick any day in the week to print", value=today, key="weekpick")
    week_start = any_day - timedelta(days=any_day.weekday())

    # Build a small 7-day schedule safely (handles month boundary)
    week_sched: dict[date, list[DeliveryItem]] = {}
    for i in range(7):
        d = week_start + timedelta(days=i)
        base_m = build_month_schedule(d.year, d.month, patients_df)
        mm_s, mm_e = month_bounds(d.year, d.month)
        ov_m = read_overrides(mm_s, mm_e)
        base_m = apply_overrides(base_m, ov_m)
        week_sched[d] = sorted(base_m.get(d, []), key=lambda x: (x.interval_weeks, x.name.lower()))

    pdf_week = make_week_pdf(week_start, week_sched, page_mode=page_mode)
    st.download_button(
        f"Download Week PDF (Landscape) — starting {week_start.isoformat()}",
        data=pdf_week,
        file_name=f"bp_week_{week_start.isoformat()}.pdf",
        mime="application/pdf",
    )

    st.divider()
    st.markdown("## Daily Delivery Sheet (BP + RX) — SDM DELIVERY SHEET LOG")

    d1, d2, d3 = st.columns([1.2, 1.0, 1.0])
    with d1:
        day_pick = st.date_input("Delivery date", value=today, key="daily_date")
    with d2:
        extra_rx = st.number_input("Extra blank RX lines", min_value=0, max_value=40, value=12, step=1)
    with d3:
        # if you want daily to be same paper setting, reuse page_mode
        pass

    # Build schedule for that month, then grab that day
    base_m = build_month_schedule(day_pick.year, day_pick.month, patients_df)
    mm_s, mm_e = month_bounds(day_pick.year, day_pick.month)
    ov_m = read_overrides(mm_s, mm_e)
    sched_m = apply_overrides(base_m, ov_m)

    day_items = sched_m.get(day_pick, [])
    day_items = sorted(day_items, key=lambda x: (x.interval_weeks, x.name.lower()))

    daily_pdf = make_daily_delivery_pdf(day_pick, day_items, extra_rx_lines=int(extra_rx), page_mode=page_mode)

    st.download_button(
        f"Download Daily Delivery Sheet — {day_pick.isoformat()}",
        data=daily_pdf,
        file_name=f"sdm_delivery_sheet_{day_pick.isoformat()}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.caption("Packages and Charged columns are intentionally blank for you to write in by hand.")
