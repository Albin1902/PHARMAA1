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
    "Auto-generates your month delivery sheet from patient frequency: Weekly / Biweekly / Monthly (4-week). "
    "Use Overrides for holidays/exceptions."
)


# =========================
# SQLite setup (KEEP SAME DB PATH)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Stability improvements:
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA busy_timeout=5000;")  # wait up to 5s instead of erroring
    return c


def table_columns(c, table_name: str) -> set[str]:
    rows = c.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}  # column name


def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                weekday INTEGER NOT NULL,                -- 0=Mon ... 6=Sun
                interval_weeks INTEGER NOT NULL,         -- 1 / 2 / 4
                packs_per_delivery INTEGER NOT NULL,     -- usually matches interval
                anchor_date TEXT NOT NULL,               -- ISO yyyy-mm-dd
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        # --- Auto-migrate: add new columns if missing (NO wipe) ---
        cols = table_columns(c, "bp_patients")

        if "address" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT")
        if "packages_per_delivery" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN packages_per_delivery INTEGER")
        if "charge_code" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN charge_code TEXT")

        # Ensure defaults for existing rows (so you don't get NULLs)
        c.execute("UPDATE bp_patients SET address = COALESCE(address, '')")
        c.execute("UPDATE bp_patients SET packages_per_delivery = COALESCE(packages_per_delivery, 1)")
        c.execute("UPDATE bp_patients SET charge_code = COALESCE(charge_code, '0')")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,                     -- ISO yyyy-mm-dd
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

CHARGE_OPTIONS = ["cc", "0"]


# =========================
# DB functions
# =========================
def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, weekday, interval_weeks, packs_per_delivery,
                   packages_per_delivery, charge_code, address,
                   anchor_date, notes, active
            FROM bp_patients
            ORDER BY active DESC, weekday ASC, interval_weeks ASC, name ASC
            """,
            c,
        )
    if df.empty:
        return df

    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)

    # normalize nullable columns
    for col in ["address", "charge_code", "notes"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "packages_per_delivery" in df.columns:
        df["packages_per_delivery"] = pd.to_numeric(df["packages_per_delivery"], errors="coerce").fillna(1).astype(int)
    if "charge_code" in df.columns:
        df.loc[~df["charge_code"].isin(CHARGE_OPTIONS), "charge_code"] = "0"

    return df


def insert_patient(
    name: str,
    weekday: int,
    interval_weeks: int,
    anchor: date,
    notes: str = "",
    address: str = "",
    packages_per_delivery: int = 1,
    charge_code: str = "0",
    active: bool = True,
):
    packs = int(interval_weeks)  # auto-fix packs to match frequency
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_patients
                (name, weekday, interval_weeks, packs_per_delivery, packages_per_delivery, charge_code, address,
                 anchor_date, notes, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                int(weekday),
                int(interval_weeks),
                packs,
                int(packages_per_delivery),
                (charge_code or "0").strip().lower(),
                address.strip(),
                anchor.isoformat(),
                notes.strip(),
                1 if active else 0,
            ),
        )
        c.commit()


def upsert_patients(df: pd.DataFrame):
    if df.empty:
        return

    df = df.copy()

    # required
    df["name"] = df["name"].astype(str).str.strip()
    df["weekday"] = df["weekday"].astype(int)
    df["interval_weeks"] = df["interval_weeks"].astype(int)
    df["packs_per_delivery"] = df["packs_per_delivery"].astype(int)
    df["active"] = df["active"].astype(bool)

    # new fields (safe defaults)
    if "address" not in df.columns:
        df["address"] = ""
    if "packages_per_delivery" not in df.columns:
        df["packages_per_delivery"] = 1
    if "charge_code" not in df.columns:
        df["charge_code"] = "0"

    df["address"] = df["address"].fillna("").astype(str)
    df["notes"] = df.get("notes", "").fillna("").astype(str)
    df["packages_per_delivery"] = pd.to_numeric(df["packages_per_delivery"], errors="coerce").fillna(1).astype(int)
    df["charge_code"] = df["charge_code"].fillna("0").astype(str).str.lower().str.strip()
    df.loc[~df["charge_code"].isin(CHARGE_OPTIONS), "charge_code"] = "0"

    # anchor_date normalize
    def _to_date(x):
        if isinstance(x, pd.Timestamp):
            return x.date()
        if isinstance(x, date):
            return x
        return pd.to_datetime(x).date()

    df["anchor_date"] = df["anchor_date"].apply(_to_date)

    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = r["name"]
            if not name:
                continue

            weekday = int(r["weekday"])
            interval = int(r["interval_weeks"])
            packs = int(r["packs_per_delivery"])
            anchor = r["anchor_date"]
            notes = str(r.get("notes", ""))
            active = 1 if bool(r["active"]) else 0

            address = str(r.get("address", "")).strip()
            packages_per_delivery = int(r.get("packages_per_delivery", 1))
            charge_code = str(r.get("charge_code", "0")).strip().lower()
            if charge_code not in CHARGE_OPTIONS:
                charge_code = "0"

            if pd.isna(rid) or rid is None:
                c.execute(
                    """
                    INSERT INTO bp_patients
                        (name, weekday, interval_weeks, packs_per_delivery, packages_per_delivery, charge_code, address,
                         anchor_date, notes, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name, weekday, interval, packs,
                        packages_per_delivery, charge_code, address,
                        anchor.isoformat(), notes, active
                    ),
                )
            else:
                c.execute(
                    """
                    UPDATE bp_patients
                    SET name=?, weekday=?, interval_weeks=?, packs_per_delivery=?,
                        packages_per_delivery=?, charge_code=?, address=?,
                        anchor_date=?, notes=?, active=?
                    WHERE id=?
                    """,
                    (
                        name, weekday, interval, packs,
                        packages_per_delivery, charge_code, address,
                        anchor.isoformat(), notes, active, int(rid)
                    ),
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
    df["note"] = df["note"].fillna("").astype(str)
    return df


def add_override(odate: date, patient_name: str, action: str, packs: int | None, note: str):
    with conn() as c:
        c.execute(
            """
            INSERT INTO bp_overrides (odate, patient_name, action, packs, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (odate.isoformat(), patient_name.strip(), action.strip(), packs, note.strip()),
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
    override_note: str = ""


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # auto schedule: weekdays only
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
        note = str(r.get("note", "") or "").strip()

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x.name != name]
        elif action == "add":
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99, override_note=note))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


def filter_schedule(schedule: dict[date, list[DeliveryItem]], mode: str) -> dict[date, list[DeliveryItem]]:
    if mode == "Weekly":
        allowed = {1, 99}
    elif mode == "Biweekly + Monthly":
        allowed = {2, 4, 99}
    else:
        allowed = {1, 2, 4, 99}

    out = {}
    for d, items in schedule.items():
        out[d] = [x for x in items if x.interval_weeks in allowed]
        out[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))
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


def wrap_to_width(text: str, max_width: float, font_name: str, font_size: int) -> list[str]:
    """Word-wrap into multiple lines within max_width."""
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
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
    left, right = margin, w - margin
    bottom, top = margin, h - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top - 0.12 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}")
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

            def max_lines_for(fs: int):
                return int(area_h // (fs + 0.5))

            chosen_mode, chosen_fs, chosen_max_lines = None, None, None

            for fs in [8, 7, 6, 5, 4, 3]:
                if fs < min_font:
                    continue
                ml = max_lines_for(fs)
                if ml <= 0:
                    continue
                if len(entries) <= ml:
                    chosen_mode, chosen_fs, chosen_max_lines = "single", fs, ml
                    break

            if chosen_mode is None and allow_two_columns:
                for fs in [7, 6, 5, 4, 3]:
                    if fs < min_font:
                        continue
                    ml = max_lines_for(fs)
                    if ml <= 0:
                        continue
                    if len(entries) <= ml * 2:
                        chosen_mode, chosen_fs, chosen_max_lines = "two", fs, ml
                        break

            if chosen_mode is None:
                chosen_mode = "two" if allow_two_columns else "single"
                chosen_fs = min_font
                chosen_max_lines = max_lines_for(chosen_fs)
                if chosen_max_lines <= 0:
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
                    to_print = entries[:cap - 1] + [f"+{remaining} more"]

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
                    to_print = entries[:cap - 1] + [f"+{remaining} more"]

                y = area_top - fs
                for i, e in enumerate(to_print):
                    line = truncate_to_width(e, half_w, font_name, fs)
                    if i < chosen_max_lines:
                        c.drawString(x_left, y - i * line_h, line)
                    else:
                        j = i - chosen_max_lines
                        c.drawString(x_right, y - j * line_h, line)

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


def make_daily_pdf(
    delivery_date: date,
    items: list[DeliveryItem],
    patients_df: pd.DataFrame,
    page_mode: str = "letter",
) -> bytes:
    """
    Daily table:
      BP (packs), Patient name, Address, Notes, Packages, Charged (cc/0)
    """
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_daily.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.35 * inch
    left, right = margin, w - margin
    bottom, top = margin, h - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top - 0.15 * inch, f"Blister Pack — Daily Delivery Sheet ({delivery_date.isoformat()})")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.40 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    table_top = top - 0.65 * inch
    table_left = left
    table_right = right
    table_width = table_right - table_left

    # Column widths (tweak here if you want)
    # BP | Name | Address | Notes | Packages | Charged
    col_fracs = [0.07, 0.18, 0.30, 0.30, 0.09, 0.06]
    col_w = [table_width * f for f in col_fracs]

    headers = ["BP", "Patient", "Address", "Notes", "Packages", "Charged"]

    # Build lookup by name (assumes unique names; if duplicates, first match used)
    lookup = {}
    if patients_df is not None and not patients_df.empty:
        for _, r in patients_df.iterrows():
            lookup[str(r["name"])] = r

    # Rows content
    rows = []
    for it in items:
        r = lookup.get(it.name, None)
        address = "" if r is None else str(r.get("address", "") or "")
        notes = "" if r is None else str(r.get("notes", "") or "")
        # if manual add override note exists, append it
        if it.override_note:
            notes = (notes + " | " + it.override_note).strip(" |")

        packages = 1 if r is None else int(r.get("packages_per_delivery", 1) or 1)
        charge = "0" if r is None else str(r.get("charge_code", "0") or "0").lower().strip()
        if charge not in CHARGE_OPTIONS:
            charge = "0"

        rows.append([str(it.packs), it.name, address, notes, str(packages), charge])

    # Drawing table
    font_name = "Helvetica"
    fs_header = 10
    fs_body = 9
    line_pad = 2

    c.setStrokeGray(0.70)
    c.setFillGray(0.92)
    header_h = 0.30 * inch
    c.rect(table_left, table_top - header_h, table_width, header_h, stroke=1, fill=1)

    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", fs_header)

    x = table_left
    for i, htxt in enumerate(headers):
        c.drawString(x + 4, table_top - header_h + 8, htxt)
        x += col_w[i]

    # vertical lines
    x = table_left
    for i in range(len(col_w) + 1):
        c.line(x, table_top, x, bottom)
        if i < len(col_w):
            x += col_w[i]

    # Row rendering (dynamic row height based on wraps)
    y = table_top - header_h
    c.setFont(font_name, fs_body)

    def cell_lines(text: str, max_w: float) -> list[str]:
        # wrap address and notes; truncate others
        if max_w <= 10:
            return [""]
        return wrap_to_width(text, max_w - 8, font_name, fs_body)

    min_row_h = 0.28 * inch

    for row in rows:
        # compute needed height
        lines_per_col = []
        for ci, val in enumerate(row):
            max_w = col_w[ci]
            if ci in (2, 3):  # address, notes wrap
                lines = cell_lines(str(val), max_w)
            else:
                lines = [truncate_to_width(str(val), max_w - 8, font_name, fs_body)]
            lines_per_col.append(lines)

        max_lines = max(len(lines) for lines in lines_per_col)
        row_h = max(min_row_h, (max_lines * (fs_body + line_pad) + 6) / 72.0 * inch)

        # page break if needed
        if y - row_h < bottom + 0.25 * inch:
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(left, top - 0.15 * inch, f"Blister Pack — Daily Delivery Sheet ({delivery_date.isoformat()})")
            c.setFont("Helvetica", 9)
            c.drawString(left, top - 0.40 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            table_top = top - 0.65 * inch

            # redraw header
            c.setStrokeGray(0.70)
            c.setFillGray(0.92)
            c.rect(table_left, table_top - header_h, table_width, header_h, stroke=1, fill=1)
            c.setFillGray(0.0)
            c.setFont("Helvetica-Bold", fs_header)
            x = table_left
            for i, htxt in enumerate(headers):
                c.drawString(x + 4, table_top - header_h + 8, htxt)
                x += col_w[i]
            # verticals
            x = table_left
            for i in range(len(col_w) + 1):
                c.line(x, table_top, x, bottom)
                if i < len(col_w):
                    x += col_w[i]
            y = table_top - header_h
            c.setFont(font_name, fs_body)

        # draw horizontal line for row
        c.setStrokeGray(0.70)
        c.line(table_left, y, table_right, y)
        c.line(table_left, y - row_h, table_right, y - row_h)

        # draw text
        x = table_left
        for ci, lines in enumerate(lines_per_col):
            yy = y - 6 - fs_body
            for ln in lines[:max_lines]:
                c.drawString(x + 4, yy, ln)
                yy -= (fs_body + line_pad)
            x += col_w[ci]

        y -= row_h

    # bottom border
    c.line(table_left, bottom, table_right, bottom)

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
            columns=[
                "id", "name", "weekday", "interval_weeks", "packs_per_delivery",
                "packages_per_delivery", "charge_code", "address",
                "anchor_date", "notes", "active"
            ]
        )
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


def export_db_bytes() -> bytes:
    with open(DB_PATH, "rb") as f:
        return f.read()


def normalize_import_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    cols = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df.rename(columns=cols, inplace=True)

    rename_map = {
        "patient_name_(printed)": "name",
        "patient_name_printed": "name",
        "patient_name": "name",
        "delivery_weekday": "weekday",
        "frequency": "interval_weeks",
        "frequency_(weeks)": "interval_weeks",
        "packs_per_delive": "packs_per_delivery",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    if "name" not in df.columns:
        raise ValueError("Import file missing column: name")
    if "weekday" not in df.columns:
        raise ValueError("Import file missing column: weekday")
    if "interval_weeks" not in df.columns:
        raise ValueError("Import file missing column: interval_weeks")
    if "anchor_date" not in df.columns:
        raise ValueError("Import file missing column: anchor_date")

    # weekday normalize
    wd_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    def _wd(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)) and str(x).strip().isdigit():
            return int(x)
        s = str(x).strip().lower()[:3]
        return wd_map.get(s, None)

    df["weekday"] = df["weekday"].apply(_wd)

    # frequency normalize
    def _freq(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s.isdigit():
            return int(s)
        if "week" in s and "bi" in s:
            return 2
        if "month" in s:
            return 4
        if "week" in s:
            return 1
        return None

    df["interval_weeks"] = df["interval_weeks"].apply(_freq)

    if "packs_per_delivery" not in df.columns:
        df["packs_per_delivery"] = df["interval_weeks"]
    df["packs_per_delivery"] = pd.to_numeric(df["packs_per_delivery"], errors="coerce").fillna(df["interval_weeks"]).astype(int)

    # new fields defaults
    if "address" not in df.columns:
        df["address"] = ""
    if "packages_per_delivery" not in df.columns:
        df["packages_per_delivery"] = 1
    if "charge_code" not in df.columns:
        df["charge_code"] = "0"

    df["address"] = df["address"].fillna("").astype(str)
    df["packages_per_delivery"] = pd.to_numeric(df["packages_per_delivery"], errors="coerce").fillna(1).astype(int)
    df["charge_code"] = df["charge_code"].fillna("0").astype(str).str.lower().str.strip()
    df.loc[~df["charge_code"].isin(CHARGE_OPTIONS), "charge_code"] = "0"

    df["anchor_date"] = pd.to_datetime(df["anchor_date"], errors="coerce").dt.date
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("").astype(str)

    if "active" not in df.columns:
        df["active"] = True
    df["active"] = df["active"].apply(lambda v: str(v).strip().lower() in ["true", "1", "yes", "y"])

    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]
    df = df[df["weekday"].isin([0, 1, 2, 3, 4])]
    df = df[df["interval_weeks"].isin([1, 2, 4])]
    df = df[df["anchor_date"].notna()]

    keep_cols = [
        "id", "name", "weekday", "interval_weeks", "packs_per_delivery",
        "packages_per_delivery", "charge_code", "address",
        "anchor_date", "notes", "active"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[keep_cols]


def import_patients(df_import: pd.DataFrame, replace: bool):
    if df_import is None or df_import.empty:
        return 0

    with conn() as c:
        if replace:
            c.execute("DELETE FROM bp_overrides")
            c.execute("DELETE FROM bp_patients")
            c.commit()

    if not replace and "id" in df_import.columns:
        df_import = df_import.copy()
        df_import["id"] = pd.NA

    upsert_patients(df_import)
    return len(df_import)


# =========================
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print + Import/Export"]
)

today = date.today()


# -------------------------
# Patients tab
# -------------------------
with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")

    with st.expander("➕ Quick Add patient (recommended)", expanded=True):
        with st.form("quick_add"):
            r1, r2, r3, r4 = st.columns([2.0, 1.1, 1.1, 1.1])
            with r1:
                new_name = st.text_input("Patient name (printed)", placeholder="e.g., Snow, Riley")
            with r2:
                new_wd = st.selectbox("Delivery weekday (Mon–Fri)", options=[0, 1, 2, 3, 4], format_func=lambda x: WEEKDAY_LABELS[int(x)])
            with r3:
                new_freq = st.selectbox("Frequency", options=[1, 2, 4], format_func=lambda v: FREQ_LABEL[int(v)])
            with r4:
                new_anchor = st.date_input("Anchor date", value=today)

            r5, r6, r7 = st.columns([2.0, 1.1, 1.0])
            with r5:
                new_address = st.text_input("Address", placeholder="Delivery address")
            with r6:
                new_packages = st.number_input("Packages", min_value=0, max_value=50, value=1, step=1)
            with r7:
                new_charge = st.selectbox("Charged", options=CHARGE_OPTIONS, index=1)

            new_notes = st.text_input("Notes (optional)", placeholder="e.g., leave at porch / call first")
            new_active = st.checkbox("Active", value=True)

            if st.form_submit_button("Add patient", type="primary"):
                if not new_name.strip():
                    st.error("Name is required.")
                else:
                    insert_patient(
                        name=new_name,
                        weekday=int(new_wd),
                        interval_weeks=int(new_freq),
                        anchor=new_anchor,
                        notes=new_notes,
                        address=new_address,
                        packages_per_delivery=int(new_packages),
                        charge_code=str(new_charge),
                        active=new_active,
                    )
                    st.success("Added.")
                    st.rerun()

    df_all = read_patients()
    if df_all.empty:
        df_all = pd.DataFrame(
            columns=[
                "id", "name", "weekday", "interval_weeks", "packs_per_delivery",
                "packages_per_delivery", "charge_code", "address",
                "anchor_date", "notes", "active"
            ]
        )

    st.divider()
    st.caption("Bulk edit below (filters affect what you see). New rows must have all required fields before saving.")

    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.2, 0.9])
    with f1:
        q = st.text_input("Search name / notes / address", value="")
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

    df_view = df_view.reset_index(drop=True)

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
            "packs_per_delivery": st.column_config.SelectboxColumn("BP (packs)", options=[1, 2, 4], required=True),
            "packages_per_delivery": st.column_config.NumberColumn("Packages", min_value=0, step=1),
            "charge_code": st.column_config.SelectboxColumn("Charged", options=CHARGE_OPTIONS),
            "address": st.column_config.TextColumn("Address"),
            "anchor_date": st.column_config.DateColumn("Anchor date", required=True),
            "notes": st.column_config.TextColumn("Notes"),
            "active": st.column_config.CheckboxColumn("Active"),
            "__delete__": st.column_config.CheckboxColumn("Delete"),
        },
    )

    auto_fix = st.toggle("Auto-fix BP packs to match frequency (recommended)", value=True)
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
    st.caption("skip removes; add inserts extra delivery (even weekends).")

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
            packs = st.number_input("BP packs", min_value=1, max_value=10, value=1, step=1)
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

    cal_obj = calendar.Calendar(firstweekday=6)
    weeks = cal_obj.monthdatescalendar(int(year), int(month))

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

            lines = [
                f"<div class='bp-item'>{it.name} <span style='opacity:0.7'>({it.packs}p)</span></div>"
                for it in shown
            ]
            more = f"<div class='bp-more'>+{extra} more</div>" if extra > 0 else ""

            cols[i].markdown(
                f"<div class='bp-cell'><div class='bp-date'>{d.day}</div>{''.join(lines)}{more}</div>",
                unsafe_allow_html=True,
            )


# -------------------------
# Print + Import/Export tab
# -------------------------
with tab_print:
    st.subheader("Print PDFs (Landscape)")

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

    st.divider()
    st.subheader("Daily Delivery PDF (table)")

    d1, d2 = st.columns([1.2, 1.8])
    with d1:
        day_pick = st.date_input("Delivery date", value=today, key="day_pick")
    with d2:
        day_scope = st.radio("Daily scope", ["All", "Weekly", "Biweekly + Monthly"], index=0, horizontal=True)

    # build schedule for that day (use its month)
    p_df = read_patients()
    base_m = build_month_schedule(day_pick.year, day_pick.month, p_df)
    ov_m = read_overrides(*month_bounds(day_pick.year, day_pick.month))
    all_m = apply_overrides(base_m, ov_m)
    filtered_m = filter_schedule(all_m, day_scope)

    day_items = filtered_m.get(day_pick, [])
    day_items = sorted(day_items, key=lambda x: (x.interval_weeks, x.name.lower()))

    if not day_items:
        st.info("No deliveries for that date (unless you add an override).")
    else:
        pdf_day = make_daily_pdf(day_pick, day_items, p_df, page_mode=page_mode)
        st.download_button(
            f"Download Daily PDF — {day_pick.isoformat()} ({day_scope})",
            data=pdf_day,
            file_name=f"bp_daily_{day_pick.isoformat()}_{day_scope.replace(' ','_').replace('+','plus').lower()}.pdf",
            mime="application/pdf",
            type="primary",
        )

    st.divider()
    st.subheader("Import / Export (restore your missing data)")

    cA, cB = st.columns([1, 1])
    with cA:
        st.download_button(
            "⬇️ Download patients CSV (backup)",
            data=export_patients_csv_bytes(),
            file_name=f"bp_patients_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    with cB:
        st.download_button(
            "⬇️ Download SQLite DB (full backup)",
            data=export_db_bytes(),
            file_name="blisterpacks.db",
            mime="application/octet-stream",
        )

    st.markdown("### Import from CSV / Excel")
    up = st.file_uploader("Upload exported file (.csv or .xlsx)", type=["csv", "xlsx"])
    replace = st.toggle("Replace existing DB (wipe then restore)", value=False)

    if up is not None:
        try:
            name = up.name.lower()
            if name.endswith(".csv"):
                df_imp = pd.read_csv(up)
            elif name.endswith(".xlsx"):
                try:
                    df_imp = pd.read_excel(up)
                except ImportError:
                    st.error("Excel import needs 'openpyxl'. Add it to requirements.txt OR save your file as real .csv and upload.")
                    st.stop()
            else:
                st.error("Unsupported file type. Use .csv or .xlsx")
                st.stop()

            df_imp2 = normalize_import_df(df_imp)

            st.write("Preview import:")
            st.dataframe(df_imp2, use_container_width=True, hide_index=True)

            if st.button("Import now", type="primary"):
                n = import_patients(df_imp2, replace=replace)
                st.success(f"Imported {n} patient rows.")
                st.rerun()

        except Exception as e:
            st.error(f"Import failed: {e}")
