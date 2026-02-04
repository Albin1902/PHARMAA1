import os
import sqlite3
import calendar
import html
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

try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:
    TZ = ZoneInfo("UTC")


def now_local() -> datetime:
    return datetime.now(TZ)


def require_pin():
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


require_pin()


# =========================
# Header
# =========================
st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption(
    "Auto-generates your delivery calendar from patient frequency: Weekly / Biweekly / Monthly (4-week). "
    "Weekdays only for auto-schedule. Use Overrides for holidays/exceptions."
)

st.warning(
    "If you print patient names, do NOT run this as a public app. Keep it private.",
    icon="⚠️",
)

# =========================
# SQLite setup (persistent path)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def safe_add_column(c: sqlite3.Connection, table: str, col_def: str):
    try:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Exception:
        pass


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
                packs_per_delivery INTEGER NOT NULL,     -- 1/2/4
                anchor_date TEXT NOT NULL,               -- ISO yyyy-mm-dd
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        safe_add_column(c, "bp_patients", "address TEXT DEFAULT ''")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,                    -- 'skip' or 'add'
                packs INTEGER,
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


def delete_patient_by_id(pid: int):
    with conn() as c:
        c.execute("DELETE FROM bp_patients WHERE id=?", (pid,))
        c.commit()


def upsert_patients(df: pd.DataFrame):
    """
    ✅ Smart upsert:
    - If id missing OR id doesn't exist in DB -> INSERT
    - Else UPDATE
    This fixes "replace (wipe) then import" even if CSV has ids.
    """
    with conn() as c:
        existing_ids = {row[0] for row in c.execute("SELECT id FROM bp_patients").fetchall()}

        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = str(r.get("name", "")).strip()
            if not name:
                continue

            address = "" if pd.isna(r.get("address", "")) else str(r.get("address", "")).strip()
            weekday = int(r.get("weekday", 0))
            interval = int(r.get("interval_weeks", 1))
            packs = int(r.get("packs_per_delivery", interval))

            anchor = r.get("anchor_date", None)
            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor, errors="coerce")
                anchor = anchor.date() if pd.notna(anchor) else date.today()

            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", "")).strip()
            active = 1 if bool(r.get("active", True)) else 0

            try:
                rid_int = int(rid) if (rid is not None and not pd.isna(rid)) else None
            except Exception:
                rid_int = None

            if rid_int is None or rid_int not in existing_ids:
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
                    (name, address, weekday, interval, packs, anchor.isoformat(), notes, active, rid_int),
                )

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


def normalize_anchor_to_weekday(anchor: date, weekday: int) -> date:
    # If anchor doesn't match weekday, push to next matching weekday
    if anchor.weekday() == weekday:
        return anchor
    delta = (weekday - anchor.weekday()) % 7
    return anchor + timedelta(days=delta)


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
    interval_weeks: int
    note: str = ""
    address: str = ""


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        if d.weekday() > 4:
            continue  # auto weekdays only

        todays = active[active["weekday"] == d.weekday()]
        if todays.empty:
            continue

        for _, r in todays.iterrows():
            raw_anchor = r["anchor_date"]
            if isinstance(raw_anchor, pd.Timestamp):
                raw_anchor = raw_anchor.date()
            if not isinstance(raw_anchor, date):
                raw_anchor = pd.to_datetime(raw_anchor, errors="coerce")
                raw_anchor = raw_anchor.date() if pd.notna(raw_anchor) else d

            interval = int(r["interval_weeks"])
            weekday = int(r["weekday"])
            anchor = normalize_anchor_to_weekday(raw_anchor, weekday)

            pt_note = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", "")).strip()
            pt_addr = "" if pd.isna(r.get("address", "")) else str(r.get("address", "")).strip()

            if occurs_on_day(anchor, interval, d):
                schedule[d].append(
                    DeliveryItem(
                        name=str(r["name"]),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=interval,
                        note=pt_note,
                        address=pt_addr,
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
        onote = "" if pd.isna(r.get("note", "")) else str(r.get("note", "")).strip()

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x.name != name]
        elif action == "add":
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99, note=onote))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
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

    margin = 0.20 * inch
    left = margin
    right = w - margin
    bottom = margin
    top = h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top - 0.12 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.30 * inch, f"Generated: {now_local().strftime('%Y-%m-%d %H:%M')}  ({TZ_NAME})")

    grid_top = top - 0.45 * inch
    grid_bottom = bottom
    grid_left = left
    grid_right = right

    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    col_w = grid_w / 7.0

    header_h = 0.22 * inch
    body_h = grid_h - header_h

    calx = calendar.Calendar(firstweekday=6)
    weeks = calx.monthdatescalendar(year, month)
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

            entries = []
            for it in items:
                s = f"{it.name} ({it.packs}p)"
                if it.note:
                    s = f"{s} — {it.note}"
                entries.append(s)

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
def export_patients_csv_bytes() -> bytes:
    df = read_patients()
    if df.empty:
        df = pd.DataFrame(
            columns=["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
        )
    return df.to_csv(index=False).encode("utf-8")


def import_patients_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Upload .csv or .xlsx")

    df.columns = [str(c).strip().lower() for c in df.columns]

    # remove junk columns like __delete__
    junk_cols = [c for c in df.columns if c.startswith("__")]
    if junk_cols:
        df = df.drop(columns=junk_cols, errors="ignore")

    required = ["name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
    for r in ["name", "weekday", "interval_weeks", "anchor_date"]:
        if r not in df.columns:
            raise ValueError(f"Missing column: {r}")

    if "address" not in df.columns:
        df["address"] = ""

    if "packs_per_delivery" not in df.columns:
        df["packs_per_delivery"] = df["interval_weeks"]

    if "active" not in df.columns:
        df["active"] = True

    if "notes" not in df.columns:
        df["notes"] = ""

    if "id" not in df.columns:
        df["id"] = pd.NA

    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").fillna(0).astype(int)
    df["interval_weeks"] = pd.to_numeric(df["interval_weeks"], errors="coerce").fillna(1).astype(int)
    df["packs_per_delivery"] = pd.to_numeric(df["packs_per_delivery"], errors="coerce").fillna(df["interval_weeks"]).astype(int)

    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["anchor_date"] = df["anchor_date"].fillna(date.today())

    def to_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("1", "true", "yes", "y", "t")

    df["active"] = df["active"].apply(to_bool)

    # Excel can put weird formula errors into notes like "#NAME?"
    df["notes"] = df["notes"].fillna("").astype(str).replace({"#NAME?": ""})
    df["address"] = df["address"].fillna("").astype(str)
    df["name"] = df["name"].astype(str).str.strip()

    df = df[["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]]
    return df


# =========================
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs"]
)

today = date.today()


# -------------------------
# Patients tab
# -------------------------
with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption("Tip: Use the Add form (fast). Use the table for bulk edits & deletes.")

    with st.expander("➕ Add new patient", expanded=True):
        with st.form("add_patient_form", clear_on_submit=True):
            c1, c2 = st.columns([2, 1.4])
            with c1:
                new_name = st.text_input("Patient name (printed)*")
            with c2:
                new_address = st.text_input("Address (optional)")

            c3, c4 = st.columns([1, 1])
            with c3:
                new_wd_label = st.selectbox("Delivery weekday (Mon–Fri)", ["Mon", "Tue", "Wed", "Thu", "Fri"], index=0)
                new_weekday = WEEKDAY_LABELS.index(new_wd_label)
            with c4:
                new_freq_label = st.selectbox("Frequency", ["Weekly", "Biweekly", "Monthly"], index=0)
                new_interval = LABEL_TO_FREQ[new_freq_label]

            c5, c6 = st.columns([1.2, 1])
            with c5:
                new_anchor = st.date_input("Anchor date (first delivery date in cycle)", value=today)
            with c6:
                new_active = st.checkbox("Active", value=True)

            new_notes = st.text_input("Notes (optional)")
            submitted = st.form_submit_button("Add patient", use_container_width=True)

            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                else:
                    packs = new_interval
                    insert_patient(
                        name=new_name,
                        address=new_address,
                        weekday=new_weekday,
                        interval_weeks=new_interval,
                        packs_per_delivery=packs,
                        anchor_date=new_anchor,
                        notes=new_notes,
                        active=new_active,
                    )
                    st.success("Patient added.")
                    st.rerun()

    st.divider()

    df_all = read_patients()
    if df_all.empty:
        df_all = pd.DataFrame(
            columns=["id", "name", "address", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
        )

    f1, f2, f3, f4 = st.columns([1.4, 1.1, 1.2, 1.0])
    with f1:
        q = st.text_input("Search name / notes / address", value="")
    with f2:
        freq_pick = st.multiselect("Frequency", ["Weekly", "Biweekly", "Monthly"], ["Weekly", "Biweekly", "Monthly"])
    with f3:
        wd_pick = st.multiselect("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri"], ["Mon", "Tue", "Wed", "Thu", "Fri"])
    with f4:
        active_only = st.toggle("Active only", value=False)

    df_view = df_all.copy()

    if q.strip():
        qq = q.strip().lower()
        df_view = df_view[
            df_view["name"].astype(str).str.lower().str.contains(qq)
            | df_view["notes"].astype(str).str.lower().str.contains(qq)
            | df_view["address"].astype(str).str.lower().str.contains(qq)
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

    edited = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Patient name (printed)", required=True),
            "address": st.column_config.TextColumn("Address"),
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

    if st.button("Save table changes", type="primary"):
        if edited.empty:
            st.info("Nothing to save.")
            st.stop()

        to_delete = edited[(edited["__delete__"] == True) & (~edited["id"].isna())]
        for _, r in to_delete.iterrows():
            delete_patient_by_id(int(r["id"]))

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

    cexp1, cexp2 = st.columns([1, 1])
    with cexp1:
        st.download_button(
            "⬇️ Download patients CSV (backup)",
            data=export_patients_csv_bytes(),
            file_name=f"bp_patients_export_{now_local().strftime('%Y-%m-%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cexp2:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button(
                    "⬇️ Download SQLite DB (full backup)",
                    data=f.read(),
                    file_name="blisterpacks.db",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

    st.markdown("### Import from CSV / Excel")
    up = st.file_uploader("Upload your exported file (.csv or .xlsx)", type=["csv", "xlsx"])
    replace_mode = st.toggle("Replace patients table (wipe then restore)", value=False)
    confirm_replace = st.checkbox("Confirm replace (required)", value=False, disabled=not replace_mode)

    if up is not None:
        try:
            df_imp = import_patients_df(up)
            st.write("Preview import:")
            st.dataframe(df_imp.head(50), use_container_width=True, hide_index=True)

            if st.button("Import now", type="primary"):
                if replace_mode and not confirm_replace:
                    st.error("You turned on Replace. Check Confirm replace first.")
                else:
                    with conn() as c:
                        if replace_mode:
                            c.execute("DELETE FROM bp_patients")
                            c.execute("DELETE FROM sqlite_sequence WHERE name='bp_patients'")
                            c.commit()

                    # ✅ after wipe, ids should NOT be used
                    if replace_mode:
                        df_imp = df_imp.copy()
                        df_imp["id"] = pd.NA

                    upsert_patients(df_imp)
                    st.success(f"Import complete. Patients now in DB: {len(read_patients())}")
                    st.rerun()

        except Exception as e:
            st.error(f"Import failed: {e}")


# -------------------------
# Overrides tab
# -------------------------
with tab_overrides:
    st.subheader("Overrides (manual exceptions)")
    st.caption("skip removes, add inserts extra delivery (even weekends).")

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
        note = st.text_input("Note", placeholder="shows on calendar/PDF if filled")

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
# Calendar tab
# -------------------------
with tab_cal:
    c1, c2, c3 = st.columns([1, 1, 1.6])
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
        view_mode = st.radio("View", ["Biweekly + Monthly", "Weekly", "All"], index=0, horizontal=True)

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
        .bp-note { font-size: 11px; opacity: 0.65; margin-left: 4px; }
        .bp-more { font-size: 12px; opacity: 0.7; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    calx = calendar.Calendar(firstweekday=6)
    weeks = calx.monthdatescalendar(int(year), int(month))

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
                note_html = ""
                if it.note:
                    note_html = f"<span class='bp-note'>— {html.escape(it.note)}</span>"
                lines.append(
                    f"<div class='bp-item'>{html.escape(it.name)} "
                    f"<span style='opacity:0.7'>({it.packs}p)</span> {note_html}</div>"
                )

            more = f"<div class='bp-more'>+{extra} more</div>" if extra > 0 else ""

            html_block = f"""
            <div class="bp-cell">
              <div class="bp-date">{d.day}</div>
              {''.join(lines)}
              {more}
            </div>
            """
            cols[i].markdown(html_block, unsafe_allow_html=True)


# -------------------------
# Print tab
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

    scope = st.radio("Month PDF scope", ["Biweekly + Monthly", "Weekly", "All"], index=0, horizontal=True)

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

    st.caption("Notes show beside patient if filled. Otherwise blank.")
