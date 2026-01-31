import os
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

st.warning(
    "If you print patient names, do NOT run this as a public app. Keep it private.",
    icon="⚠️"
)

# =========================
# SQLite setup
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
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    return df


def upsert_patients(df: pd.DataFrame):
    with conn() as c:
        for _, r in df.iterrows():
            rid = r.get("id", None)
            name = str(r["name"]).strip()
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
    interval_weeks: int  # 1 weekly, 2 biweekly, 4 monthly


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}
    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in list(schedule.keys()):
        # Deliveries weekdays only (Mon-Fri). Overrides can still add manually if you want.
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

        # ✅ Sort by frequency: weekly (1) → biweekly (2) → monthly (4), then name
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
            # Manual "add" gets interval=99 so it shows after monthly (keeps your weekly/bi/month order clean)
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99))
            schedule[d].sort(key=lambda x: (x.interval_weeks, x.name.lower()))

    return schedule


# =========================
# PDF helpers
# =========================
def wrap_text_to_width(text: str, max_width: float, font_name: str, font_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def make_month_pdf_one_page(
    year: int,
    month: int,
    schedule: dict[date, list[DeliveryItem]],
    page_mode: str = "letter",   # "letter" or "legal"
    max_shrink: int = 3,         # minimum font size allowed
    allow_two_columns: bool = True,
) -> bytes:
    """
    ✅ ONE PAGE ONLY month calendar
    ✅ Uses tight margins + dynamic font shrinking
    ✅ Can use 2-column packing inside a day cell to fit more names
    ✅ Shows ONLY dates for the selected month (other-month cells blank)
    ✅ Order: weekly → biweekly → monthly already sorted in schedule
    """

    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize
    tmp_path = os.path.join(DATA_DIR, "_tmp_month_onepage.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    # Tight layout
    margin = 0.25 * inch
    left = margin
    right = w - margin
    bottom = margin
    top = h - margin

    # Small header (uses less space)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, top - 0.15 * inch, f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}")
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.35 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Grid area
    grid_top = top - 0.55 * inch
    grid_bottom = bottom
    grid_left = left
    grid_right = right

    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom

    col_w = grid_w / 7.0

    header_h = 0.22 * inch
    body_h = grid_h - header_h

    # Weeks for THIS month only (4-6 rows depending on month)
    cal = calendar.Calendar(firstweekday=6)  # Sunday-first
    weeks = cal.monthdatescalendar(year, month)
    rows = len(weeks)  # 4..6
    row_h = body_h / rows

    # Header background
    c.setFillGray(0.92)
    c.rect(grid_left, grid_top - header_h, grid_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 9)
    for i, lbl in enumerate(SUN_FIRST):
        c.drawString(grid_left + i * col_w + 3, grid_top - header_h + 6, lbl)

    # Borders
    c.setStrokeGray(0.70)
    c.rect(grid_left, grid_bottom, grid_w, grid_h, stroke=1, fill=0)
    c.line(grid_left, grid_top - header_h, grid_right, grid_top - header_h)

    # Determine worst-day load so we can pick a font that fits ALL cells
    # We also allow per-cell two-column fallback.
    font_name = "Helvetica"

    def build_entry_strings(items: list[DeliveryItem]) -> list[str]:
        # packs display like (2p) and keep order already sorted
        out = []
        for it in items:
            out.append(f"{it.name} ({it.packs}p)")
        return out

    # Render each cell
    for r, week in enumerate(weeks):
        y_top = grid_top - header_h - r * row_h
        y_bot = y_top - row_h

        # row line
        c.setStrokeGray(0.70)
        c.line(grid_left, y_top, grid_right, y_top)

        for col, d in enumerate(week):
            x0 = grid_left + col * col_w
            x1 = x0 + col_w

            # col line
            c.setStrokeGray(0.70)
            c.line(x0, y_bot, x0, y_top)

            # Other-month cells: blank (no date, no names)
            if d.month != month:
                continue

            # Date number
            c.setFont("Helvetica-Bold", 9)
            c.drawString(x0 + 3, y_top - 12, str(d.day))

            items = schedule.get(d, [])
            entries = build_entry_strings(items)

            # Text area inside cell
            pad_x = 3
            pad_y_top = 18  # below date
            pad_y_bottom = 3
            area_top = y_top - pad_y_top
            area_bottom = y_bot + pad_y_bottom
            area_h = max(0, area_top - area_bottom)

            if not entries or area_h <= 0:
                continue

            # Try to fit single-column, then optional 2-column
            def try_single(fs: int):
                line_h = fs + 1
                max_lines = int(area_h // line_h)
                if max_lines <= 0:
                    return None

                max_w = (col_w - 2 * pad_x)
                lines = []
                for e in entries:
                    wrapped = wrap_text_to_width(e, max_w, font_name, fs)
                    lines.extend(wrapped)
                if len(lines) <= max_lines:
                    return ("single", fs, lines, max_lines)
                return None

            def try_two_col(fs: int):
                if not allow_two_columns:
                    return None
                line_h = fs + 1
                max_lines = int(area_h // line_h)
                if max_lines <= 0:
                    return None

                gap = 6
                half_w = (col_w - 2 * pad_x - gap) / 2.0
                # wrap each entry to half width
                all_lines = []
                for e in entries:
                    wrapped = wrap_text_to_width(e, half_w, font_name, fs)
                    all_lines.extend(wrapped)

                if len(all_lines) <= max_lines * 2:
                    return ("two", fs, all_lines, max_lines)
                return None

            chosen = None
            # shrink until it fits
            for fs in [8, 7, 6, 5, 4, max_shrink]:
                fs = int(fs)
                if fs < max_shrink:
                    continue
                chosen = try_single(fs)
                if chosen:
                    break

            if not chosen:
                # if single can’t fit, try 2-column packing
                for fs in [7, 6, 5, 4, max_shrink]:
                    fs = int(fs)
                    if fs < max_shrink:
                        continue
                    chosen = try_two_col(fs)
                    if chosen:
                        break

            # If STILL not fitting at min font: we render as much as possible (still one page)
            if not chosen:
                fs = max_shrink
                line_h = fs + 1
                max_lines = int(area_h // line_h)
                gap = 6
                half_w = (col_w - 2 * pad_x - gap) / 2.0 if allow_two_columns else (col_w - 2 * pad_x)

                if allow_two_columns:
                    # two-column hard cap
                    c.setFont(font_name, fs)
                    x_left = x0 + pad_x
                    x_right = x0 + pad_x + half_w + gap
                    y = area_top - fs
                    cap = max_lines * 2
                    lines = []
                    for e in entries:
                        lines.extend(wrap_text_to_width(e, half_w, font_name, fs))
                    lines = lines[:cap]
                    # fill left column then right
                    for i, line in enumerate(lines):
                        if i < max_lines:
                            c.drawString(x_left, y - i * line_h, line)
                        else:
                            j = i - max_lines
                            c.drawString(x_right, y - j * line_h, line)
                else:
                    # single-column hard cap
                    c.setFont(font_name, fs)
                    max_w = col_w - 2 * pad_x
                    lines = []
                    for e in entries:
                        lines.extend(wrap_text_to_width(e, max_w, font_name, fs))
                    lines = lines[:max_lines]
                    y = area_top - fs
                    for i, line in enumerate(lines):
                        c.drawString(x0 + pad_x, y - i * line_h, line)

                continue

            mode, fs, lines, max_lines = chosen
            c.setFont(font_name, fs)
            line_h = fs + 1

            if mode == "single":
                y = area_top - fs
                for i, line in enumerate(lines[:max_lines]):
                    c.drawString(x0 + pad_x, y - i * line_h, line)

            else:  # two columns
                gap = 6
                half_w = (col_w - 2 * pad_x - gap) / 2.0
                x_left = x0 + pad_x
                x_right = x0 + pad_x + half_w + gap
                y = area_top - fs

                cap = max_lines * 2
                lines = lines[:cap]

                for i, line in enumerate(lines):
                    if i < max_lines:
                        c.drawString(x_left, y - i * line_h, line)
                    else:
                        j = i - max_lines
                        c.drawString(x_right, y - j * line_h, line)

    # right border line and bottom line
    c.setStrokeGray(0.70)
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
    c.drawString(0.5 * inch, h - 0.8 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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
            wrapped = wrap_text_to_width(line, max_w, "Helvetica", 9)
            for wline in wrapped:
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
# UI Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs"]
)

today = date.today()


with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption("Interval (weeks): 1=Weekly, 2=Biweekly, 4=Monthly (4-week). Anchor date defines the cycle start.")

    df = read_patients()
    if df.empty:
        df = pd.DataFrame(
            columns=["id", "name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
        )

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.TextColumn("Patient name (printed)", required=True),
            "weekday": st.column_config.SelectboxColumn(
                "Delivery weekday (Mon–Fri)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: WEEKDAY_LABELS[int(x)],
                required=True,
            ),
            "interval_weeks": st.column_config.SelectboxColumn("Frequency (weeks)", options=[1, 2, 4], required=True),
            "packs_per_delivery": st.column_config.SelectboxColumn("Packs per delivery", options=[1, 2, 4], required=True),
            "anchor_date": st.column_config.DateColumn("Anchor date", required=True),
            "notes": st.column_config.TextColumn("Notes"),
            "active": st.column_config.CheckboxColumn("Active"),
        },
    )

    auto_fix = st.toggle("Auto-fix packs to match frequency (recommended)", value=True)
    if auto_fix and not edited.empty:
        for idx in edited.index:
            try:
                interval = int(edited.loc[idx, "interval_weeks"])
                edited.loc[idx, "packs_per_delivery"] = interval  # 1/2/4
            except Exception:
                pass

    if st.button("Save changes", type="primary"):
        bad = edited[edited["name"].astype(str).str.strip() == ""]
        if not bad.empty:
            st.error("Some rows have empty names. Fix them or delete those rows.")
            st.stop()
        upsert_patients(edited)
        st.success("Saved.")
        st.rerun()


with tab_overrides:
    st.subheader("Overrides (manual exceptions)")
    st.caption("Use overrides for holidays/patient-specific changes: skip removes, add inserts extra delivery.")

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
        note = st.text_input("Note", placeholder="e.g., holiday shift / patient ok pickup")

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
        st.dataframe(ov, use_container_width=True)
        del_id = st.number_input("Override ID to delete", min_value=1, step=1)
        confirm = st.checkbox("Confirm delete override")
        if st.button("Delete override", disabled=not confirm):
            delete_override(int(del_id))
            st.success("Deleted override.")
            st.rerun()


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
    schedule = apply_overrides(base, overrides_df)

    st.subheader(f"{calendar.month_name[int(month)]} {int(year)} (Sun → Sat)")

    st.markdown(
        """
        <style>
        .bp-cell { border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 8px; min-height: 140px; }
        .bp-date { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
        .bp-muted { opacity: 0.25; }
        .bp-item { font-size: 12px; line-height: 1.25; margin: 0 0 2px 0; }
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

            shown = items[:12]
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

    allow_two_cols = st.toggle("Max fit (use 2 columns inside busy days)", value=True)
    min_font = st.slider("Minimum font size (smaller = fits more)", min_value=3, max_value=6, value=3)

    patients_df = read_patients()
    base = build_month_schedule(int(py), int(pm), patients_df)
    start, end = month_bounds(int(py), int(pm))
    overrides_df = read_overrides(start, end)
    sched = apply_overrides(base, overrides_df)

    pdf_month = make_month_pdf_one_page(
        int(py),
        int(pm),
        sched,
        page_mode=page_mode,
        max_shrink=int(min_font),
        allow_two_columns=allow_two_cols,
    )

    st.download_button(
        "Download Month PDF (ONE PAGE, Landscape)",
        data=pdf_month,
        file_name=f"bp_month_{py}_{pm:02d}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.divider()
    st.markdown("### Weekly PDF (optional)")
    any_day = st.date_input("Pick any day in the week to print", value=today, key="weekpick")
    week_start = any_day - timedelta(days=any_day.weekday())

    week_sched: dict[date, list[DeliveryItem]] = {}
    for i in range(7):
        d = week_start + timedelta(days=i)
        base_m = build_month_schedule(d.year, d.month, patients_df)
        ov_m = read_overrides(*month_bounds(d.year, d.month))
        base_m = apply_overrides(base_m, ov_m)
        week_sched[d] = sorted(base_m.get(d, []), key=lambda x: (x.interval_weeks, x.name.lower()))

    pdf_week = make_week_pdf(week_start, week_sched, page_mode=page_mode)
    st.download_button(
        f"Download Week PDF (Landscape) — starting {week_start.isoformat()}",
        data=pdf_week,
        file_name=f"bp_week_{week_start.isoformat()}.pdf",
        mime="application/pdf",
    )
