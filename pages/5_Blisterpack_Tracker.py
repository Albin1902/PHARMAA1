import os
import io
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

# Timezone (fixes "Generated" stamp time on Streamlit Cloud which is often UTC)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback


# =========================================================
# Page config + PIN lock
# =========================================================
st.set_page_config(page_title="Blister Pack Delivery Sheet", layout="wide")

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
    st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


# =========================================================
# Header
# =========================================================
st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption(
    "Auto-generates your month delivery sheet from patient frequency: Weekly / Biweekly / Monthly (4-week). "
    "Use Overrides for holidays/exceptions."
)

# =========================================================
# App timezone (Generated timestamp)
# Set in Streamlit secrets if you want:
# BP_TIMEZONE = "America/Toronto"
# =========================================================
APP_TZ = str(st.secrets.get("BP_TIMEZONE", "America/Toronto"))
def now_local() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(APP_TZ))
    except Exception:
        return datetime.now()


# =========================================================
# SQLite setup (KEEP SAME DB PATH so your data doesn't vanish)
# =========================================================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)

def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _ensure_column(c: sqlite3.Connection, table: str, col: str, ddl_fragment: str):
    cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {ddl_fragment}")

def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT,
                weekday INTEGER NOT NULL,                -- 0=Mon ... 6=Sun
                interval_weeks INTEGER NOT NULL,         -- 1 / 2 / 4  (Weekly / Biweekly / Monthly)
                packs_per_delivery INTEGER NOT NULL,     -- usually matches interval (1/2/4)
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

        # Migrations (safe if table already existed)
        _ensure_column(c, "bp_patients", "address", "address TEXT")

        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_patients_weekday ON bp_patients(weekday)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_overrides_date ON bp_overrides(odate)")
        c.commit()

init_db()

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SUN_FIRST = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
FREQ_LABEL = {1: "Weekly", 2: "Biweekly", 4: "Monthly"}
LABEL_TO_FREQ = {"Weekly": 1, "Biweekly": 2, "Monthly": 4}


# =========================================================
# DB functions
# =========================================================
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
    df["active"] = df["active"].astype(int).astype(bool)
    df["address"] = df["address"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df

def insert_patient(name: str, address: str, weekday: int, interval: int, packs: int, anchor: date, notes: str, active: bool):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_patients (name, address, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name.strip(), address.strip(), int(weekday), int(interval), int(packs), anchor.isoformat(), notes.strip(), 1 if active else 0),
        )
        c.commit()

def upsert_patients(df: pd.DataFrame):
    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = str(r.get("name", "")).strip()
            address = "" if pd.isna(r.get("address", "")) else str(r.get("address", "")).strip()
            weekday = int(r["weekday"])
            interval = int(r["interval_weeks"])
            packs = int(r["packs_per_delivery"])

            anchor = r["anchor_date"]
            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor).date()

            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))
            active = 1 if bool(r["active"]) else 0

            if not name:
                continue

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

def clean_old_prefixes():
    """
    If your DB accidentally has names like "w John" / "B Jane" / "M Bob", remove those.
    """
    with conn() as c:
        rows = c.execute("SELECT id, name, interval_weeks FROM bp_patients").fetchall()
        for pid, nm, interval in rows:
            if not isinstance(nm, str):
                continue
            s = nm.strip()
            # remove only a leading single-letter prefix + space
            if len(s) >= 2 and s[1] == " " and s[0].lower() in ("w", "b", "m"):
                new_name = s[2:].strip()
                if new_name:
                    c.execute("UPDATE bp_patients SET name=? WHERE id=?", (new_name, pid))
        c.commit()


# =========================================================
# Scheduling logic
# =========================================================
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
    address: str = ""

def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # automatic deliveries weekdays only
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
                        address=str(r.get("address", "") or ""),
                    )
                )

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
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99, address=""))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return schedule

def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
    """
    mode:
      - "Weekly" -> {1, 99}
      - "Biweekly + Monthly" -> {2, 4, 99}
      - "All" -> {1,2,4,99}
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


# =========================================================
# PDF helpers
# =========================================================
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
)
