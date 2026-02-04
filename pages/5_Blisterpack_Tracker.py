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
TZ_NAME = str(st.secrets.get("BP_TZ", "America/Toronto"))

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
    "Weekdays only (Mon–Fri) for automation. Use Overrides for holidays/exceptions."
)

# =========================
# SQLite setup  (KEEP SAME DB PATH)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)

def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _has_column(c: sqlite3.Connection, table: str, col: str) -> bool:
    rows = c.execute(f"PRAGMA table_info({table});").fetchall()
    cols = {r[1] for r in rows}
    return col in cols

def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT DEFAULT '',
                weekday INTEGER NOT NULL,                -- 0=Mon ... 6=Sun
                interval_weeks INTEGER NOT NULL,         -- 1 / 2 / 4
                packs_per_delivery INTEGER NOT NULL,     -- usually 1/2/4
                anchor_date TEXT NOT NULL,               -- ISO yyyy-mm-dd
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,                     -- ISO date yyyy-mm-dd
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,                    -- 'skip' or 'add'
                packs INTEGER,                           -- used for 'add'
                note TEXT
            )
            """
        )

        # lightweight migrations (if older DB exists)
        if not _has_column(c, "bp_patients", "address"):
            c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT DEFAULT ''")

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
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    # address/notes can be null in old rows
    df["address"] = df["address"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df


def upsert_patients(df: pd.DataFrame):
    """
    IMPORTANT: If id exists -> UPDATE.
    If UPDATE affects 0 rows (e.g., table was wiped), we INSERT instead.
    This prevents the “wipe then restore shows nothing” issue.
    """
    if df.empty:
        return

    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = str(r.get("name", "")).strip()
            address = "" if pd.isna(r.get("address", "")) else str(r.get("address", "")).strip()
            weekday = int(r.get("weekday", 0))
            interval = int(r.get("interval_weeks", 1))
            packs = int(r.get("packs_per_delivery", interval))
            anchor = r.get("anchor_date", None)

            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor).date() if anchor else date.today()

            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))
            active = 1 if bool(r.get("active", True)) else 0

            if not name:
                continue

            # normalize ranges
            weekday = max(0, min(6, weekday))
            if interval not in (1, 2, 4):
                interval = 1
            if packs not in (1, 2, 4):
                packs = interval

            # Update then fallback insert
            if rid is not None and not pd.isna(rid):
                cur = c.execute(
                    """
                    UPDATE bp_patients
                    SET name=?, address=?, weekday=?, interval_weeks=?, packs_per_delivery=?, anchor_date=?, notes=?, active=?
                    WHERE id=?
                    """,
                    (name, address, weekday, interval, packs, anchor.isoformat(), notes, active, int(rid)),
                )
                if cur.rowcount == 0:
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
                    INSERT INTO bp_patients (name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, address, weekday, interval, packs, anchor.isoformat(), notes, active),
                )

        c.commit()


def delete_patient_by_id(pid: int):
    with conn() as c:
        c.execute("DELETE FROM bp_patients WHERE id=?", (pid,))
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
    df["odate"] = pd.to_datetime(df["odate"]).dt.date
    df["note"] = df["note"].fillna("")
    return df


def add_override(odate: date, patient_name: str, action: str, packs: int | None, note: str):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_overrides (odate, patient_name, action, packs, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (odate.isoformat(), patient_name.strip(), action, packs, note.strip()),
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
    name: str
    packs: int
    interval_weeks: int  # 1 weekly, 2 biweekly, 4 monthly, 99 manual add


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # weekdays only for automatic schedule
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
                        name=str(r["name"]),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=interval,
                    )
                )

        # Sort: weekly → biweekly → monthly → manual(99), then name
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

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x.name != name]
        elif action == "add":
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
    """
    mode:
      - "Biweekly + Monthly" -> intervals {2,4} + manual 99
      - "Weekly" -> {1} + manual 99
      - "All" -> {1,2,4} + manual 99
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
# PDF helpers (month one-page)
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
    page_mode: str = "letter",
    min_font: int = 3,
    allow_two_columns: bool = True,
) -> bytes:
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_month_onepage.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    tz = ZoneInfo(TZ_NAME)
    generated = datetime.now(tz)

    margin = 0.20 * inch
    left = margin
    right = w - margin
    bottom = margin
    top = h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top - 0.12 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.30 * inch, f"Generated: {generated.strftime('%Y-%m-%d %H:%M')}")

    grid_top = top - 0.45 * inch
    grid_bottom = bottom
    grid_left = left
    grid_right = right

    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    col_w = grid_w / 7.0

    header_h = 0.22 * inch
    body_h = grid_h - header_h

    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(year, month)
    rows = len(weeks)
    row_h = body_h / rows

    c.setFillGray(0.92)
    c.rect(grid_left, grid_top - header_h, grid_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 9)
    for i, lbl in enumerate(SUN_FIRST):
        c.drawString(grid_left + i * col_w + 3, grid_top - header_h + 6, lbl)

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

            if d.month != month:
                continue

            c.setFont("Helvetica-Bold", 9)
            c.drawString(x0 + 3, y_top - 12, str(d.day))

            items = schedule.get(d, [])
            if not items:
                continue

            pad_x = 3
            pad_y_top = 18
            pad_y_bottom = 3
            area_top = y_top - pad_y_top
            area_bottom = y_bot + pad_y_bottom
            area_h = max(0, area_top - area_bottom)
            if area_h <= 0:
                continue

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
                    line = truncate_to_width(e, max_w, font_name, fs)
                    c.drawString(x0 + pad_x, y - i * line_h, line)

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
# Import / Export helpers
# =========================
EXPECTED_COLS = ["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]

def normalize_import_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts CSV/XLSX with any subset of columns.
    Fixes:
      - weekday can be 0..4 OR 1..5 (Mon..Fri) -> normalize to 0..4
      - active can be TRUE/FALSE/1/0
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # common aliases
    rename_map = {
        "patient": "name",
        "patient_name": "name",
        "week_day": "weekday",
        "frequency": "interval_weeks",
        "packs": "packs_per_delivery",
        "pack_per_delivery": "packs_per_delivery",
        "anchor": "anchor_date",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # ensure missing columns exist
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    def wd_to_int(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (int, float)) and str(x).strip() != "":
            v = int(x)
            # If user used Mon=1..Fri=5 convert to 0..4
            if 1 <= v <= 5:
                return v - 1
            # If already 0..4 keep
            if 0 <= v <= 4:
                return v
            return 0

        s = str(x).strip().title()[:3]
        if s in WEEKDAY_LABELS:
            return WEEKDAY_LABELS.index(s)
        return 0

    df["weekday"] = df["weekday"].apply(wd_to_int).astype(int)

    df["interval_weeks"] = pd.to_numeric(df["interval_weeks"], errors="coerce").fillna(1).astype(int)
    df["interval_weeks"] = df["interval_weeks"].apply(lambda v: v if v in (1, 2, 4) else 1)

    df["packs_per_delivery"] = pd.to_numeric(df["packs_per_delivery"], errors="coerce")
    df["packs_per_delivery"] = df.apply(
        lambda r: int(r["interval_weeks"]) if pd.isna(r["packs_per_delivery"]) else int(r["packs_per_delivery"]),
        axis=1,
    )
    df["packs_per_delivery"] = df["packs_per_delivery"].apply(lambda v: v if v in (1, 2, 4) else 1).astype(int)

    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["anchor_date"] = df["anchor_date"].fillna(date.today())

    df["name"] = df["name"].astype(str).fillna("").apply(lambda s: s.strip())
    df["address"] = df["address"].astype(str).fillna("").apply(lambda s: s.strip())
    df["notes"] = df["notes"].astype(str).fillna("").apply(lambda s: s.strip())

    def to_bool(x):
        if pd.isna(x):
            return True
        s = str(x).strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
        return True

    df["active"] = df["active"].apply(to_bool)

    # id: keep numeric if present, else NA
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df[EXPECTED_COLS]


def read_upload(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()

    # Try best-guess based on extension
    try_csv_first = name.endswith(".csv") or uploaded.type in ("text/csv", "application/vnd.ms-excel")

    raw = None
    if try_csv_first:
        raw = pd.read_csv(uploaded)
    else:
        # .xlsx import requires openpyxl (in requirements)
        try:
            raw = pd.read_excel(uploaded)
        except Exception:
            # fallback: sometimes people have weird names like export.csv.xlsx
            uploaded.seek(0)
            raw = pd.read_csv(uploaded)

    return normalize_import_df(raw)


def export_patients_csv(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


# =========================
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs"]
)

today = date.today()


# -------------------------
# Patients tab (filters + ADD FORM + delete + import/export)
# -------------------------
with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption("Frequency controls automation. Anchor date defines the cycle start. Weekday should be Mon–Fri.")

    df_all = read_patients()
    if df_all.empty:
        df_all = pd.DataFrame(columns=EXPECTED_COLS)

    with st.expander("➕ Add new patient (recommended)", expanded=True):
        a1, a2, a3, a4 = st.columns([2.2, 2.2, 1.2, 1.2])
        with a1:
            new_name = st.text_input("Patient name (printed)", value="", key="new_name")
            new_addr = st.text_input("Address (optional)", value="", key="new_addr")
        with a2:
            new_notes = st.text_input("Notes (optional)", value="", key="new_notes")
            new_anchor = st.date_input("Anchor date", value=today, key="new_anchor")
        with a3:
            new_wd = st.selectbox("Delivery weekday", options=[0,1,2,3,4], format_func=lambda x: WEEKDAY_LABELS[x], key="new_wd")
            new_freq = st.selectbox("Frequency", options=[1,2,4], format_func=lambda x: FREQ_LABEL[x], key="new_freq")
        with a4:
            new_active = st.checkbox("Active", value=True, key="new_active")
            auto_packs = True
            new_packs = new_freq  # always match

        if st.button("Add patient", type="primary"):
            if not new_name.strip():
                st.error("Name is required.")
            else:
                to_add = pd.DataFrame([{
                    "id": pd.NA,
                    "name": new_name.strip(),
                    "address": new_addr.strip(),
                    "weekday": int(new_wd),
                    "interval_weeks": int(new_freq),
                    "packs_per_delivery": int(new_packs),
                    "anchor_date": new_anchor,
                    "notes": new_notes.strip(),
                    "active": bool(new_active),
                }])
                upsert_patients(to_add)
                st.success("Added.")
                st.rerun()

    st.divider()

    # Filters
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.3, 1.0])
    with f1:
        q = st.text_input("Search name / address / notes", value="")
    with f2:
        freq_pick = st.multiselect(
            "Frequency filter",
            options=["Weekly", "Biweekly", "Monthly"],
            default=["Weekly", "Biweekly", "Monthly"],
        )
    with f3:
        wd_pick = st.multiselect(
            "Weekday filter",
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

    # editor + delete flag
    df_edit = df_view.copy()
    if "__delete__" not in df_edit.columns:
        df_edit["__delete__"] = False

    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Patient name (printed)", required=True),
            "address": st.column_config.TextColumn("Address", width="large"),
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

        # delete rows first
        to_delete = edited[(edited["__delete__"] == True) & (~edited["id"].isna())]
        for _, r in to_delete.iterrows():
            delete_patient_by_id(int(r["id"]))

        # upsert remaining
        keep = edited[edited["__delete__"] == False].drop(columns=["__delete__"], errors="ignore")

        bad = keep[keep["name"].astype(str).str.strip() == ""]
        if not bad.empty:
            st.error("Some rows have empty names. Fix them or delete those rows.")
            st.stop()

        upsert_patients(keep)
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.subheader("Import / Export (restore your missing data)")

    # Export buttons
    ex1, ex2 = st.columns([1, 1])
    with ex1:
        st.download_button(
            "⬇️ Download patients CSV (backup)",
            data=export_patients_csv(read_patients()),
            file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_patients_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        # download sqlite file
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                "⬇️ Download SQLite DB (full backup)",
                data=db_bytes,
                file_name="blisterpacks.db",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.info("DB file not found yet (no data).")

    st.markdown("### Import from CSV / Excel")
    uploaded = st.file_uploader("Upload your exported file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

    replace_db = st.toggle("Replace existing patients (wipe then restore)", value=False)
    if uploaded is not None:
        try:
            imp = read_upload(uploaded)
            st.success(f"Loaded file with {len(imp)} rows.")
            st.dataframe(imp.head(25), use_container_width=True, hide_index=True)

            if st.button("Import now", type="primary"):
                if replace_db:
                    with conn() as c:
                        c.execute("DELETE FROM bp_patients;")
                        c.commit()
                    # safest: force inserts
                    imp["id"] = pd.NA

                upsert_patients(imp)
                st.success(f"Imported {len(imp)} rows.")
                st.rerun()

        except Exception as e:
            st.error(f"Import failed: {e}")


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
    oc1, oc2, oc3, oc4, oc5 = st.columns([1.1, 1.7, 1.2, 0.9, 1.4])
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
    c1, c2, c3 = st.columns([1, 1, 1.8])
    with c1:
        year = st.number_input("Year", min_value=2020, max_value=2100, value=today.year, step=1)
    with c2:
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: calendar.month_name[m],
        )
    with c3:
        view_mode = st.radio(
            "View",
            ["Biweekly + Monthly", "Weekly", "All"],
            index=0,
            horizontal=True,
        )

    patients_df = read_patients()
    base = build_month_schedule(int(year), int(month), patients_df)
    start, end = month_bounds(int(year), int(month))
    overrides_df = read_overrides(start, end)
    schedule_all = apply_overrides(base, overrides_df)
    schedule = filter_schedule(schedule_all, view_mode)

    st.subheader(f"{calendar.month_name[int(month)]} {int(year)} — {view_mode}")

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


# -------------------------
# Print tab
# -------------------------
with tab_print:
    st.subheader("Print Month PDF (Landscape • One Page • Selected Month Only)")

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

    scope = st.radio(
        "Month PDF scope",
        ["Biweekly + Monthly", "Weekly", "All"],
        index=0,
        horizontal=True,
    )

    allow_two_cols = st.toggle("Max fit (use 2 columns inside busy days)", value=True)
    min_font = st.slider("Minimum font size (smaller = fits more)", min_value=3, max_value=6, value=3)

    patients_df = read_patients()
    base = build_month_schedule(int(py), int(pm), patients_df)
    mstart, mend = month_bounds(int(py), int(pm))
    overrides_df = read_overrides(mstart, mend)
    sched_all = apply_overrides(base, overrides_df)
    sched = filter_schedule(sched_all, scope)

    pdf_month = make_month_pdf_one_page(
        int(py),
        int(pm),
        sched,
        page_mode=page_mode,
        min_font=int(min_font),
        allow_two_columns=allow_two_cols,
    )

    fname_scope = scope.replace(" ", "_").replace("+", "plus").lower()
    st.download_button(
        f"Download Month PDF — {scope} (ONE PAGE, Landscape)",
        data=pdf_month,
        file_name=f"bp_month_{py}_{pm:02d}_{fname_scope}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.caption("If a cell has too many names, the PDF shows “+X more” instead of silently dropping them.")
