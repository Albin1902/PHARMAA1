import os
import io
import re
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


# =========================
# Page config + PIN lock
# =========================
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


# =========================
# Header
# =========================
st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption(
    "Auto-generates a month delivery calendar from patient frequency: Weekly / Biweekly / Monthly (4-week). "
    "Weekdays only for automatic scheduling. Use Overrides for holidays/exceptions."
)

# =========================
# SQLite setup  (KEEP SAME DB PATH so your data doesn't vanish)
# =========================
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

        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_patients_weekday ON bp_patients(weekday)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bp_overrides_date ON bp_overrides(odate)")
        c.commit()


init_db()

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SUN_FIRST = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

FREQ_LABEL = {1: "Weekly", 2: "Biweekly", 4: "Monthly"}
LABEL_TO_FREQ = {"Weekly": 1, "Biweekly": 2, "Monthly": 4}


# =========================
# Small helpers
# =========================
def clean_display_name(name: str) -> str:
    """
    Removes the weird prefix that you saw in PDF/UI:
      "W ", "B ", "M " (upper/lower) at the start of a name.
    This fixes display/printing even if old bad names exist in DB.
    """
    if not isinstance(name, str):
        name = str(name)
    n = name.strip()
    # only remove single-letter + space prefixes
    if len(n) >= 2 and n[1] == " " and n[0] in ("W", "w", "B", "b", "M", "m"):
        return n[2:].strip()
    return n


def bool_from_any(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


# =========================
# DB functions
# =========================
def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
            FROM bp_patients
            ORDER BY active DESC, weekday ASC, interval_weeks ASC, name ASC
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["active"] = df["active"].astype(int).astype(bool)
    return df


def upsert_patients(df: pd.DataFrame):
    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)

            name = str(r.get("name", "")).strip()
            if not name:
                continue

            weekday = int(r.get("weekday", 0))
            interval = int(r.get("interval_weeks", 1))
            packs = int(r.get("packs_per_delivery", interval))
            anchor = r.get("anchor_date", date.today())

            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor, errors="coerce").date()

            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))
            active = 1 if bool_from_any(r.get("active", True)) else 0

            # basic clamps
            if weekday < 0:
                weekday = 0
            if weekday > 6:
                weekday = 6
            if interval not in (1, 2, 4):
                interval = 1
            if packs < 1:
                packs = 1

            if pd.isna(rid) or rid is None:
                c.execute(
                    """
                    INSERT INTO bp_patients (name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, weekday, interval, packs, anchor.isoformat(), notes, active),
                )
            else:
                c.execute(
                    """
                    UPDATE bp_patients
                    SET name=?, weekday=?, interval_weeks=?, packs_per_delivery=?, anchor_date=?, notes=?, active=?
                    WHERE id=?
                    """,
                    (name, weekday, interval, packs, anchor.isoformat(), notes, active, int(rid)),
                )
        c.commit()


def insert_patients_replace_all(df: pd.DataFrame):
    """
    Wipes patient table then inserts rows. If IDs exist, preserves them.
    """
    with conn() as c:
        c.execute("DELETE FROM bp_patients")
        # also wipe sqlite autoincrement sequence
        c.execute("DELETE FROM sqlite_sequence WHERE name='bp_patients'")
        c.commit()

    # insert with explicit id if present
    with conn() as c:
        for _, r in df.iterrows():
            name = str(r.get("name", "")).strip()
            if not name:
                continue
            weekday = int(r.get("weekday", 0))
            interval = int(r.get("interval_weeks", 1))
            packs = int(r.get("packs_per_delivery", interval))
            anchor = r.get("anchor_date", date.today())
            if isinstance(anchor, pd.Timestamp):
                anchor = anchor.date()
            if not isinstance(anchor, date):
                anchor = pd.to_datetime(anchor, errors="coerce").date()
            notes = "" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))
            active = 1 if bool_from_any(r.get("active", True)) else 0

            rid = r.get("id", None)
            if rid is None or pd.isna(rid):
                c.execute(
                    """
                    INSERT INTO bp_patients (name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, weekday, interval, packs, anchor.isoformat(), notes, active),
                )
            else:
                c.execute(
                    """
                    INSERT INTO bp_patients (id, name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (int(rid), name, weekday, interval, packs, anchor.isoformat(), notes, active),
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
    df["odate"] = pd.to_datetime(df["odate"], errors="coerce").dt.date
    return df


def add_override(odate: date, patient_name: str, action: str, packs: int | None, note: str):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_overrides (odate, patient_name, action, packs, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (odate.isoformat(), patient_name.strip(), action, packs, (note or "").strip()),
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
    out = []
    d = start
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
                        name=clean_display_name(str(r["name"])),
                        packs=int(r["packs_per_delivery"]),
                        interval_weeks=interval,
                    )
                )

        # Sort: weekly → biweekly → monthly, then name
        schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


def apply_overrides(schedule: dict[date, list[DeliveryItem]], overrides_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    if overrides_df.empty:
        return schedule

    for _, r in overrides_df.iterrows():
        d = r["odate"]
        name = clean_display_name(str(r["patient_name"]))
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if clean_display_name(x.name) != name]
        elif action == "add":
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
    """
    mode:
      - "Weekly" -> intervals {1} plus manual 99
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
# PDF helpers (ONE PAGE month, landscape)
# =========================
def truncate_to_width(text: str, max_width: float, font_name: str, font_size: int) -> str:
    """Single-line truncation with ellipsis so we don't wrap + silently drop names."""
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
    title_suffix: str,
    page_mode: str = "letter",        # "letter" or "legal"
    min_font: int = 3,
    allow_two_columns: bool = True,
) -> bytes:
    """
    ✅ ONE PAGE ONLY
    ✅ Selected month ONLY (other-month cells blank, and NO extra date numbers)
    ✅ Landscape
    ✅ Truncates lines and uses "+X more" if still too busy
    """
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
    c.drawString(left, top - 0.12 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year} — {title_suffix}")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.30 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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

            # BLANK other-month cells completely (no date number)
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

            entries = [f"{clean_display_name(it.name)} ({it.packs}p)" for it in items]

            def max_lines_for(fs: int) -> int:
                return int(area_h // (fs + 0.5))

            chosen_mode = None
            chosen_fs = None
            chosen_lines = None

            for fs in [8, 7, 6, 5, 4, 3]:
                if fs < min_font:
                    continue
                ml = max_lines_for(fs)
                if ml <= 0:
                    continue

                # single column fit?
                if len(entries) <= ml:
                    chosen_mode, chosen_fs, chosen_lines = "single", fs, ml
                    break

            if chosen_mode is None and allow_two_columns:
                for fs in [7, 6, 5, 4, 3]:
                    if fs < min_font:
                        continue
                    ml = max_lines_for(fs)
                    if ml <= 0:
                        continue
                    if len(entries) <= ml * 2:
                        chosen_mode, chosen_fs, chosen_lines = "two", fs, ml
                        break

            if chosen_mode is None:
                chosen_mode = "two" if allow_two_columns else "single"
                chosen_fs = min_font
                chosen_lines = max_lines_for(chosen_fs)
                if chosen_lines <= 0:
                    continue

            fs = chosen_fs
            c.setFont(font_name, fs)
            line_h = fs + 0.5

            if chosen_mode == "single":
                max_w = col_w - 2 * pad_x
                cap = chosen_lines
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

                cap = chosen_lines * 2
                to_print = entries[:cap]
                remaining = len(entries) - len(to_print)

                if remaining > 0 and cap >= 1:
                    to_print = entries[:cap - 1]
                    to_print.append(f"+{remaining} more")

                y = area_top - fs
                for i, e in enumerate(to_print):
                    line = truncate_to_width(e, half_w, font_name, fs)
                    if i < chosen_lines:
                        c.drawString(x_left, y - i * line_h, line)
                    else:
                        j = i - chosen_lines
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
# Import/Export (CSV works without openpyxl)
# =========================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


def normalize_import_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts exported CSV columns and also a few variations.
    Required: name, weekday, interval_weeks, packs_per_delivery, anchor_date, active
    notes optional, id optional
    """
    # normalize column names
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # handle common variations
    rename_map = {
        "weekd": "weekday",
        "week_day": "weekday",
        "interval_week": "interval_weeks",
        "interval": "interval_weeks",
        "packs_per_delive": "packs_per_delivery",
        "packs_per_deliv": "packs_per_delivery",
        "packs": "packs_per_delivery",
        "anchor_d": "anchor_date",
        "anchor": "anchor_date",
        "act": "active",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # minimal required columns
    needed = ["name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "active"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Your CSV must include: {needed}")

    # coerce types
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce")

    df["name"] = df["name"].astype(str).str.strip()
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").fillna(0).astype(int)
    df["interval_weeks"] = pd.to_numeric(df["interval_weeks"], errors="coerce").fillna(1).astype(int)
    df["packs_per_delivery"] = pd.to_numeric(df["packs_per_delivery"], errors="coerce").fillna(df["interval_weeks"]).astype(int)

    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    df["active"] = df["active"].apply(bool_from_any)

    if "notes" not in df.columns:
        df["notes"] = ""
    else:
        df["notes"] = df["notes"].fillna("").astype(str)

    # clamp values
    df.loc[df["weekday"] < 0, "weekday"] = 0
    df.loc[df["weekday"] > 6, "weekday"] = 6
    df.loc[~df["interval_weeks"].isin([1, 2, 4]), "interval_weeks"] = 1
    df.loc[df["packs_per_delivery"] < 1, "packs_per_delivery"] = 1
    df.loc[df["anchor_date"].isna(), "anchor_date"] = date.today()

    # keep only relevant cols
    keep_cols = ["id", "name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep_cols]
    return df


def read_uploaded_as_csv(uploaded) -> pd.DataFrame:
    """
    Reads CSV regardless of filename. This avoids openpyxl entirely.
    """
    raw = uploaded.getvalue()
    # try utf-8 first, then latin-1
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            # quick sanity: must have commas + header "name"
            if "," in text.splitlines()[0].lower():
                df = pd.read_csv(io.StringIO(text))
                return df
        except Exception:
            pass
    # final attempt: pandas can sometimes read bytes directly
    return pd.read_csv(io.BytesIO(raw))


# =========================
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print, tab_io = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs", "📥 Import/Export"]
)

today = date.today()


# -------------------------
# Patients tab (filters + delete)
# -------------------------
with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption("Frequency controls automation. Anchor date defines the cycle start.")

    df_all = read_patients()
    if df_all.empty:
        df_all = pd.DataFrame(
            columns=["id", "name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
        )

    f1, f2, f3, f4 = st.columns([1.4, 1.1, 1.2, 1.0])
    with f1:
        q = st.text_input("Search name / notes", value="")
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

    edited = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Patient name (printed)", required=True),
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

        keep = edited[edited["__delete__"] == False].drop(columns=["__delete__"], errors="ignore")

        bad = keep[keep["name"].astype(str).str.strip() == ""]
        if not bad.empty:
            st.error("Some rows have empty names. Fix them or delete those rows.")
            st.stop()

        upsert_patients(keep)
        st.success("Saved.")
        st.rerun()


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
    names = sorted(patients_df["name"].map(clean_display_name).tolist()) if not patients_df.empty else []

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
            "Calendar view",
            ["Biweekly + Monthly", "Weekly", "All"],
            index=0,  # default
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
        .bp-cell { border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 8px; min-height: 150px; }
        .bp-date { font-weight: 800; font-size: 14px; margin-bottom: 6px; }
        .bp-muted { opacity: 0.18; }
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
            items = sorted(items, key=lambda x: (x.interval_weeks, x.name.lower()))

            shown = items[:14]
            extra = max(0, len(items) - len(shown))

            lines = []
            for it in shown:
                nm = clean_display_name(it.name)
                lines.append(f"<div class='bp-item'>{nm} <span style='opacity:0.7'>({it.packs}p)</span></div>")

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
# Print tab (download matches Weekly vs Biweekly+Monthly)
# -------------------------
with tab_print:
    st.subheader("Print Month PDF (Landscape • ONE PAGE • Selected Month Only)")

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
        "Download",
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
        title_suffix=scope,
        page_mode=page_mode,
        min_font=int(min_font),
        allow_two_columns=allow_two_cols,
    )

    fname_scope = scope.replace(" ", "_").replace("+", "plus").lower()
    st.download_button(
        f"Download Month PDF — {scope} (ONE PAGE)",
        data=pdf_month,
        file_name=f"bp_month_{py}_{pm:02d}_{fname_scope}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.caption("If a day is too crowded, the PDF shows “+X more” instead of silently dropping names.")


# -------------------------
# Import / Export tab (restore your missing data)
# -------------------------
with tab_io:
    st.subheader("Import / Export (restore your missing data)")

    df_now = read_patients()
    if df_now.empty:
        df_now = pd.DataFrame(columns=["id", "name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"])

    cexp1, cexp2 = st.columns([1, 1])
    with cexp1:
        st.download_button(
            "⬇️ Download patients CSV (backup)",
            data=df_to_csv_bytes(df_now),
            file_name=f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_patients_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cexp2:
        # full DB backup
        try:
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                "⬇️ Download SQLite DB (full backup)",
                data=db_bytes,
                file_name="blisterpacks.db",
                mime="application/x-sqlite3",
                use_container_width=True,
            )
        except Exception:
            st.info("DB file not found yet (it will appear after you save at least one patient).")

    st.divider()

    st.markdown("### Import from CSV (recommended)")
    st.caption(
        "Your upload failed earlier because your file was **.xlsx** which needs **openpyxl**. "
        "Upload a real **.csv** here (Excel → Save As → CSV)."
    )

    uploaded = st.file_uploader("Upload exported patients file", type=["csv", "txt"])
    replace_all = st.toggle("Replace existing DB (wipe then restore)", value=False)

    if uploaded is not None:
        try:
            raw_df = read_uploaded_as_csv(uploaded)
            imp = normalize_import_df(raw_df)
            st.success(f"Loaded {len(imp)} rows from upload.")
            st.dataframe(imp.head(30), use_container_width=True, hide_index=True)

            if st.button("Import now", type="primary"):
                if replace_all:
                    insert_patients_replace_all(imp)
                    st.success("Imported (replaced existing patient table).")
                else:
                    upsert_patients(imp)
                    st.success("Imported (merged into existing table).")
                st.rerun()

        except Exception as e:
            st.error(f"Import failed: {e}")

