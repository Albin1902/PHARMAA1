import os
import io
import sqlite3
import calendar as pycal
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics


# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Blister Pack Delivery Sheet", layout="wide")

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "blisterpacks.db")

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WD_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}

FREQ_LABEL = {1: "Weekly", 2: "Biweekly", 4: "Monthly (4-week)"}
FREQ_ORDER = [1, 2, 4]

# Use secrets if present, else default to 2026 (your request)
REQUIRED_PIN = str(st.secrets.get("BP_PIN", "2026"))


# ============================================================
# DB Helpers
# ============================================================
def ensure_db():
    os.makedirs(DB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                weekday TEXT NOT NULL,                 -- Mon..Fri
                interval_weeks INTEGER NOT NULL,       -- 1 / 2 / 4
                packs_per_delivery INTEGER NOT NULL,   -- 1 / 2 / 4 etc
                anchor_date TEXT NOT NULL,             -- YYYY-MM-DD (a known delivery date)
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                action TEXT NOT NULL,                  -- skip | move | add
                src_date TEXT NOT NULL,                -- YYYY-MM-DD
                dst_date TEXT,                         -- YYYY-MM-DD (for move)
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
            """
        )
        conn.commit()


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_patients(active_only: bool = False) -> List[dict]:
    q = """
        SELECT id, name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
        FROM patients
    """
    args = []
    if active_only:
        q += " WHERE active = 1"
    q += " ORDER BY weekday, name"
    with get_conn() as conn:
        rows = conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def add_patient(
    name: str,
    weekday: str,
    interval_weeks: int,
    packs_per_delivery: int,
    anchor_date: str,
    notes: str,
    active: int,
):
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO patients (name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, weekday, int(interval_weeks), int(packs_per_delivery), anchor_date, notes, int(active), ts, ts),
        )
        conn.commit()


def update_patient(
    patient_id: int,
    name: str,
    weekday: str,
    interval_weeks: int,
    packs_per_delivery: int,
    anchor_date: str,
    notes: str,
    active: int,
):
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE patients
            SET name=?, weekday=?, interval_weeks=?, packs_per_delivery=?, anchor_date=?, notes=?, active=?, updated_at=?
            WHERE id=?
            """,
            (name, weekday, int(interval_weeks), int(packs_per_delivery), anchor_date, notes, int(active), ts, int(patient_id)),
        )
        conn.commit()


def delete_patient(patient_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM overrides WHERE patient_id=?", (int(patient_id),))
        conn.execute("DELETE FROM patients WHERE id=?", (int(patient_id),))
        conn.commit()


def fetch_overrides_for_month(month_start: date, month_end: date) -> List[dict]:
    # Pull overrides where src_date or dst_date are in the month (dst_date may be null)
    ms = month_start.isoformat()
    me = month_end.isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT o.id, o.patient_id, o.action, o.src_date, o.dst_date, o.note, p.name
            FROM overrides o
            JOIN patients p ON p.id = o.patient_id
            WHERE (o.src_date BETWEEN ? AND ?) OR (o.dst_date BETWEEN ? AND ?)
            ORDER BY o.src_date ASC
            """,
            (ms, me, ms, me),
        ).fetchall()
    return [dict(r) for r in rows]


def add_override(patient_id: int, action: str, src_date: str, dst_date: Optional[str], note: str):
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO overrides (patient_id, action, src_date, dst_date, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(patient_id), action, src_date, dst_date, note, ts),
        )
        conn.commit()


def delete_override(override_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM overrides WHERE id=?", (int(override_id),))
        conn.commit()


# ============================================================
# Scheduling Logic
# ============================================================
@dataclass
class DeliveryItem:
    patient_id: int
    name: str
    interval_weeks: int  # 1/2/4
    packs: int
    note: str = ""


def month_bounds(any_day: date) -> Tuple[date, date]:
    start = date(any_day.year, any_day.month, 1)
    last_day = pycal.monthrange(any_day.year, any_day.month)[1]
    end = date(any_day.year, any_day.month, last_day)
    return start, end


def normalize_anchor(anchor: date, desired_weekday: str) -> date:
    """Ensure anchor matches the chosen weekday (Mon-Fri). If not, shift forward to next matching weekday."""
    if desired_weekday not in WD_TO_IDX:
        return anchor
    target = WD_TO_IDX[desired_weekday]
    delta = (target - anchor.weekday()) % 7
    return anchor + timedelta(days=delta)


def iter_occurrences(anchor: date, interval_weeks: int, start: date, end: date):
    step_days = 7 * int(interval_weeks)
    step = timedelta(days=step_days)

    # Find first occurrence >= start
    if anchor < start:
        delta_days = (start - anchor).days
        k = (delta_days + step_days - 1) // step_days  # ceil
        cur = anchor + k * step
    else:
        cur = anchor

    while cur <= end:
        yield cur
        cur += step


def build_override_maps(overrides: List[dict]):
    skips = set()  # (patient_id, src_date)
    moves = {}     # (patient_id, src_date) -> dst_date
    adds = []      # (patient_id, src_date)

    for o in overrides:
        pid = int(o["patient_id"])
        action = o["action"]
        src = o["src_date"]
        dst = o.get("dst_date")

        if action == "skip":
            skips.add((pid, src))
        elif action == "move" and dst:
            moves[(pid, src)] = dst
        elif action == "add":
            adds.append((pid, src))

    return skips, moves, adds


def generate_month_schedule(selected_month: date, freq_filter: Optional[int] = None) -> Dict[date, List[DeliveryItem]]:
    start, end = month_bounds(selected_month)
    patients = fetch_patients(active_only=True)
    overrides = fetch_overrides_for_month(start, end)
    skips, moves, adds = build_override_maps(overrides)

    schedule: Dict[date, List[DeliveryItem]] = {}

    def add_item(d: date, item: DeliveryItem):
        if d < start or d > end:
            return
        schedule.setdefault(d, []).append(item)

    # Base occurrences from patient rules
    for p in patients:
        interval = int(p["interval_weeks"])
        if freq_filter is not None and interval != int(freq_filter):
            continue

        weekday = p["weekday"]
        anchor = date.fromisoformat(p["anchor_date"])
        anchor = normalize_anchor(anchor, weekday)

        for occ in iter_occurrences(anchor, interval, start, end):
            src_key = (int(p["id"]), occ.isoformat())

            # Apply skip / move
            if src_key in skips:
                continue
            if src_key in moves:
                dst = date.fromisoformat(moves[src_key])
                add_item(
                    dst,
                    DeliveryItem(
                        patient_id=int(p["id"]),
                        name=p["name"],
                        interval_weeks=interval,
                        packs=int(p["packs_per_delivery"]),
                        note=str(p.get("notes") or ""),
                    ),
                )
                continue

            add_item(
                occ,
                DeliveryItem(
                    patient_id=int(p["id"]),
                    name=p["name"],
                    interval_weeks=interval,
                    packs=int(p["packs_per_delivery"]),
                    note=str(p.get("notes") or ""),
                ),
            )

    # Extra adds
    if adds:
        pid_to_patient = {int(p["id"]): p for p in patients}
        for pid, src_date in adds:
            if pid not in pid_to_patient:
                continue
            p = pid_to_patient[pid]
            interval = int(p["interval_weeks"])
            if freq_filter is not None and interval != int(freq_filter):
                continue
            d = date.fromisoformat(src_date)
            add_item(
                d,
                DeliveryItem(
                    patient_id=int(p["id"]),
                    name=p["name"],
                    interval_weeks=interval,
                    packs=int(p["packs_per_delivery"]),
                    note=str(p.get("notes") or ""),
                ),
            )

    # Sort inside each day: Weekly first -> Biweekly -> Monthly, then by name
    for d in schedule:
        schedule[d].sort(key=lambda x: (FREQ_ORDER.index(x.interval_weeks) if x.interval_weeks in FREQ_ORDER else 99, x.name.lower()))

    return schedule


# ============================================================
# Calendar Rendering (Streamlit)
# ============================================================
def build_month_grid(year: int, month: int) -> List[List[Optional[int]]]:
    cal = pycal.Calendar(firstweekday=6)  # Sunday start
    weeks = cal.monthdayscalendar(year, month)  # 0 for outside month
    # Always show 6 rows for consistent layout
    while len(weeks) < 6:
        weeks.append([0] * 7)
    return [[d if d != 0 else None for d in w] for w in weeks]


def render_calendar_html(selected_month: date, schedule: Dict[date, List[DeliveryItem]], title: str, show_totals: bool):
    y, m = selected_month.year, selected_month.month
    weeks = build_month_grid(y, m)

    st.markdown(
        """
        <style>
        .bp-cal { display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; }
        .bp-head { font-weight: 700; opacity: 0.9; padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.08); }
        .bp-cell { border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 8px; min-height: 120px; background: rgba(255,255,255,0.02); }
        .bp-date { font-weight: 800; font-size: 14px; opacity: 0.95; }
        .bp-lines { margin-top: 6px; font-size: 12px; line-height: 1.25; opacity: 0.92; }
        .bp-muted { opacity: 0.55; }
        .bp-total { font-size: 12px; opacity: 0.8; margin-left: 6px; }
        .bp-tag { font-size: 10px; opacity: 0.65; margin-left: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(title)

    # Headers
    headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    head_html = '<div class="bp-cal">' + "".join([f'<div class="bp-head">{h}</div>' for h in headers]) + "</div>"
    st.markdown(head_html, unsafe_allow_html=True)

    # Cells
    cells = []
    for w in weeks:
        for day in w:
            if day is None:
                cells.append('<div class="bp-cell bp-muted"></div>')
                continue

            d = date(y, m, day)
            items = schedule.get(d, [])
            total_packs = sum(i.packs for i in items)

            top = f'<div><span class="bp-date">{day}</span>'
            if show_totals and items:
                top += f'<span class="bp-total">(Total: {total_packs}p)</span>'
            top += "</div>"

            lines = []
            for it in items:
                # Short tag: W / B / M
                tag = "W" if it.interval_weeks == 1 else ("B" if it.interval_weeks == 2 else ("M" if it.interval_weeks == 4 else str(it.interval_weeks)))
                lines.append(f'{it.name} ({it.packs}p)<span class="bp-tag">{tag}</span>')
            lines_html = "<div class='bp-lines'>" + "<br/>".join(lines) + "</div>" if lines else "<div class='bp-lines bp-muted'> </div>"

            cells.append(f'<div class="bp-cell">{top}{lines_html}</div>')

    cal_html = '<div class="bp-cal">' + "".join(cells) + "</div>"
    st.markdown(cal_html, unsafe_allow_html=True)


# ============================================================
# PDF Generation (Single page, landscape, month-only)
# ============================================================
def wrap_line(text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    """Greedy wrap by words using ReportLab stringWidth."""
    if not text:
        return [""]
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        cand = w if not cur else (cur + " " + w)
        if pdfmetrics.stringWidth(cand, font_name, font_size) <= max_width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # Handle single very-long token (no spaces)
    fixed = []
    for ln in lines:
        if pdfmetrics.stringWidth(ln, font_name, font_size) <= max_width:
            fixed.append(ln)
        else:
            # hard cut with ellipsis
            s = ln
            while s and pdfmetrics.stringWidth(s + "…", font_name, font_size) > max_width:
                s = s[:-1]
            fixed.append((s + "…") if s else "…")
    return fixed


def build_month_pdf(selected_month: date, freq_filter: Optional[int], show_totals: bool) -> bytes:
    schedule = generate_month_schedule(selected_month, freq_filter=freq_filter)
    start, end = month_bounds(selected_month)

    # Page setup
    page_w, page_h = landscape(A4)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))

    # Layout
    margin = 28
    title_h = 40
    grid_top = page_h - margin - title_h
    grid_left = margin
    grid_right = page_w - margin
    grid_bottom = margin

    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom

    cols = 7
    rows = 6  # always 6 rows for month grid
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    # Title
    month_name = selected_month.strftime("%B %Y")
    filt_label = "All" if freq_filter is None else FREQ_LABEL.get(int(freq_filter), str(freq_filter))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - margin - 8, f"Blister Pack Delivery Sheet — {month_name} ({filt_label})")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawString(margin, page_h - margin - 26, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.setFillColor(colors.black)

    # Headers (Sun..Sat)
    headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    header_font = 10
    c.setFont("Helvetica-Bold", header_font)
    for i, h in enumerate(headers):
        x = grid_left + i * cell_w + 6
        c.drawString(x, grid_top + 8, h)

    # Grid lines
    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(0.8)
    # outer border
    c.rect(grid_left, grid_bottom, grid_w, grid_h, stroke=1, fill=0)
    # verticals
    for i in range(1, cols):
        x = grid_left + i * cell_w
        c.line(x, grid_bottom, x, grid_top)
    # horizontals
    for j in range(1, rows):
        y = grid_bottom + j * cell_h
        c.line(grid_left, y, grid_right, y)

    # Month grid (Sunday-first) with blank cells outside month
    weeks = build_month_grid(selected_month.year, selected_month.month)

    # Prepare all cell contents first to estimate needed font size
    font_name = "Helvetica"
    max_font = 9
    min_font = 6

    cell_contents: Dict[Tuple[int, int], List[str]] = {}  # (r,c) -> lines
    for r in range(rows):
        for col in range(cols):
            day = weeks[r][col]
            if day is None:
                cell_contents[(r, col)] = []
                continue
            d = date(selected_month.year, selected_month.month, day)
            items = schedule.get(d, [])
            lines = []
            # Date line is drawn separately; lines are deliveries
            for it in items:
                tag = "W" if it.interval_weeks == 1 else ("B" if it.interval_weeks == 2 else ("M" if it.interval_weeks == 4 else str(it.interval_weeks)))
                lines.append(f"{it.name} ({it.packs}p) [{tag}]")
            cell_contents[(r, col)] = lines

    # Compute a font size that fits worst-case cell
    usable_h = cell_h - 18  # allow date at top
    usable_w = cell_w - 12
    chosen_font = max_font

    # We need to account for wrapping; estimate by wrapping at chosen_font
    def total_wrapped_lines(lines: List[str], fs: int) -> int:
        n = 0
        for ln in lines:
            n += len(wrap_line(ln, usable_w, font_name, fs))
        return n

    for fs in range(max_font, min_font - 1, -1):
        worst = 0
        for key, lines in cell_contents.items():
            worst = max(worst, total_wrapped_lines(lines, fs))
        # allow totals line too if enabled
        extra = 1 if show_totals else 0
        needed = worst + extra
        # each line height ~ fs + 1
        if needed == 0:
            chosen_font = fs
            break
        if (needed * (fs + 1)) <= usable_h:
            chosen_font = fs
            break

    # Draw each cell content
    for r in range(rows):
        for col in range(cols):
            day = weeks[r][col]
            x0 = grid_left + col * cell_w
            # PDF y origin is bottom; row 0 is top row
            y1 = grid_top - r * cell_h
            y0 = y1 - cell_h

            if day is None:
                continue

            d = date(selected_month.year, selected_month.month, day)
            items = schedule.get(d, [])

            # Date number (top-left)
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(colors.black)
            c.drawString(x0 + 6, y1 - 14, str(day))

            # Optional total packs
            y_cursor = y1 - 26
            if show_totals and items:
                total_packs = sum(i.packs for i in items)
                c.setFont("Helvetica", 8)
                c.setFillColor(colors.grey)
                c.drawString(x0 + 6, y_cursor, f"Total: {total_packs} packs")
                c.setFillColor(colors.black)
                y_cursor -= 10

            # Lines
            c.setFont(font_name, chosen_font)
            wrapped_all = []
            for it in items:
                tag = "W" if it.interval_weeks == 1 else ("B" if it.interval_weeks == 2 else ("M" if it.interval_weeks == 4 else str(it.interval_weeks)))
                base = f"{it.name} ({it.packs}p) [{tag}]"
                wrapped_all.extend(wrap_line(base, usable_w, font_name, chosen_font))

            # Fit to cell; truncate if too many
            max_lines = int(max(0, usable_h) // (chosen_font + 1))
            if show_totals and items:
                # totals used some space already, reduce max_lines a bit
                max_lines = int(max(0, (usable_h - 10) // (chosen_font + 1)))

            to_draw = wrapped_all[:max_lines]
            remaining = len(wrapped_all) - len(to_draw)
            if remaining > 0 and max_lines > 0:
                # replace last line with "+N more"
                to_draw[-1] = f"+ {remaining} more"

            for ln in to_draw:
                if y_cursor <= y0 + 6:
                    break
                c.drawString(x0 + 6, y_cursor, ln)
                y_cursor -= (chosen_font + 1)

    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================
# PIN Lock
# ============================================================
def pin_gate():
    st.sidebar.markdown("### 🔒 Blister Pack Tracker Lock")
    if "bp_unlocked" not in st.session_state:
        st.session_state.bp_unlocked = False

    pin = st.sidebar.text_input("Enter PIN", type="password")
    c1, c2 = st.sidebar.columns([1, 1])
    with c1:
        if st.button("Unlock", use_container_width=True):
            if str(pin).strip() == REQUIRED_PIN:
                st.session_state.bp_unlocked = True
                st.success("Unlocked.")
            else:
                st.session_state.bp_unlocked = False
                st.error("Wrong PIN.")
    with c2:
        if st.button("Lock", use_container_width=True):
            st.session_state.bp_unlocked = False
            st.info("Locked.")

    if not st.session_state.bp_unlocked:
        st.warning("Locked. Enter PIN to continue.")
        st.stop()


# ============================================================
# UI: Patients Tab (Filter/Edit/Delete)
# ============================================================
def patients_tab():
    st.subheader("Patients master list (Filter / Edit / Delete)")

    patients = fetch_patients(active_only=False)

    # Filters
    c1, c2, c3, c4 = st.columns([1.8, 1.2, 1.2, 0.8])
    with c1:
        q = st.text_input("Search (name / notes)", placeholder="e.g., snow, sensors")
    with c2:
        freq_sel = st.multiselect(
            "Frequency",
            options=[1, 2, 4],
            default=[],
            format_func=lambda x: FREQ_LABEL.get(int(x), str(x)),
        )
    with c3:
        wd_sel = st.multiselect("Weekday", options=WEEKDAYS, default=[])
    with c4:
        active_only = st.toggle("Active only", value=True)

    # Apply filters (pure python)
    def matches(p: dict) -> bool:
        if active_only and int(p["active"]) != 1:
            return False
        if freq_sel and int(p["interval_weeks"]) not in [int(x) for x in freq_sel]:
            return False
        if wd_sel and p["weekday"] not in wd_sel:
            return False
        if q.strip():
            qq = q.strip().lower()
            hay = f"{p.get('name','')} {p.get('notes','')}".lower()
            if qq not in hay:
                return False
        return True

    filtered = [p for p in patients if matches(p)]
    # Sort weekday then name
    filtered.sort(key=lambda p: (WD_TO_IDX.get(p["weekday"], 99), str(p["name"]).lower()))

    # Display (compact)
    if not filtered:
        st.info("No matches. Adjust filters.")
    else:
        # make a lightweight table (no pandas required)
        rows = []
        for p in filtered:
            rows.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "weekday": p["weekday"],
                    "frequency": FREQ_LABEL.get(int(p["interval_weeks"]), p["interval_weeks"]),
                    "packs": p["packs_per_delivery"],
                    "anchor": p["anchor_date"],
                    "active": "✅" if int(p["active"]) == 1 else "—",
                    "notes": p.get("notes", "") or "",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.divider()

        def label(p: dict) -> str:
            return f"#{p['id']} — {p['name']} ({p['weekday']} / {FREQ_LABEL.get(int(p['interval_weeks']), p['interval_weeks'])})"

        selected = st.selectbox("Select a patient to edit", options=filtered, format_func=label)

        e1, e2, e3 = st.columns([1.6, 1.1, 1.1])
        with e1:
            new_name = st.text_input("Patient name (printed)", value=str(selected["name"]))
            new_notes = st.text_area("Notes", value=str(selected.get("notes") or ""), height=90)
        with e2:
            new_weekday = st.selectbox("Delivery weekday (Mon–Fri)", WEEKDAYS, index=WEEKDAYS.index(selected["weekday"]))
            new_interval = st.selectbox(
                "Frequency",
                options=[1, 2, 4],
                index=[1, 2, 4].index(int(selected["interval_weeks"])),
                format_func=lambda x: FREQ_LABEL.get(int(x), str(x)),
            )
            new_active = st.checkbox("Active", value=bool(int(selected["active"])))
        with e3:
            new_packs = st.number_input("Packs per delivery", min_value=1, value=int(selected["packs_per_delivery"]), step=1)
            new_anchor = st.date_input("Anchor date", value=date.fromisoformat(selected["anchor_date"]))

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Save changes", type="primary", use_container_width=True):
                update_patient(
                    patient_id=int(selected["id"]),
                    name=new_name.strip(),
                    weekday=new_weekday,
                    interval_weeks=int(new_interval),
                    packs_per_delivery=int(new_packs),
                    anchor_date=new_anchor.isoformat(),
                    notes=new_notes.strip(),
                    active=1 if new_active else 0,
                )
                st.success("Saved.")
                st.rerun()

        with b2:
            if st.button("Delete patient", use_container_width=True):
                delete_patient(int(selected["id"]))
                st.warning("Deleted.")
                st.rerun()

    st.divider()

    # Add new
    with st.expander("➕ Add new patient"):
        a1, a2, a3 = st.columns([1.6, 1.1, 1.1])
        with a1:
            a_name = st.text_input("Name", key="add_name")
            a_notes = st.text_area("Notes", key="add_notes", height=90)
        with a2:
            a_weekday = st.selectbox("Weekday", WEEKDAYS, key="add_weekday")
            a_interval = st.selectbox("Frequency", [1, 2, 4], format_func=lambda x: FREQ_LABEL[int(x)], key="add_interval")
            a_active = st.checkbox("Active", value=True, key="add_active")
        with a3:
            a_packs = st.number_input("Packs per delivery", min_value=1, value=1, step=1, key="add_packs")
            a_anchor = st.date_input("Anchor date", key="add_anchor", value=date.today())

        if st.button("Add patient", use_container_width=True):
            if not a_name.strip():
                st.error("Name required.")
            else:
                add_patient(
                    name=a_name.strip(),
                    weekday=a_weekday,
                    interval_weeks=int(a_interval),
                    packs_per_delivery=int(a_packs),
                    anchor_date=a_anchor.isoformat(),
                    notes=a_notes.strip(),
                    active=1 if a_active else 0,
                )
                st.success("Added.")
                st.rerun()


# ============================================================
# UI: Overrides Tab
# ============================================================
def overrides_tab(selected_month: date):
    st.subheader("Overrides (Skip / Move / Add)")

    start, end = month_bounds(selected_month)

    patients = fetch_patients(active_only=True)
    if not patients:
        st.info("Add patients first.")
        return

    label_map = {p["id"]: f"{p['name']} (#{p['id']})" for p in patients}

    c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.2, 1.6])
    with c1:
        pid = st.selectbox("Patient", options=[p["id"] for p in patients], format_func=lambda x: label_map[int(x)])
    with c2:
        action = st.selectbox("Action", options=["skip", "move", "add"])
    with c3:
        src = st.date_input("Date", value=start)
    with c4:
        dst = None
        if action == "move":
            dst = st.date_input("Move to", value=src + timedelta(days=1))

    note = st.text_input("Note (optional)", placeholder="e.g., holiday, patient requested change")

    if st.button("Add override", type="primary"):
        add_override(
            patient_id=int(pid),
            action=action,
            src_date=src.isoformat(),
            dst_date=(dst.isoformat() if (action == "move" and dst) else None),
            note=note.strip(),
        )
        st.success("Override added.")
        st.rerun()

    st.divider()

    existing = fetch_overrides_for_month(start, end)
    if not existing:
        st.info("No overrides for this month.")
        return

    for o in existing:
        with st.container(border=True):
            st.write(
                f"**#{o['id']}** — **{o['name']}** | **{o['action'].upper()}** | {o['src_date']}"
                + (f" → {o['dst_date']}" if o.get("dst_date") else "")
            )
            if o.get("note"):
                st.caption(o["note"])
            if st.button("Delete override", key=f"del_ov_{o['id']}"):
                delete_override(int(o["id"]))
                st.warning("Deleted.")
                st.rerun()


# ============================================================
# UI: Calendar Tab (split views)
# ============================================================
def calendar_tab(selected_month: date):
    st.subheader(f"Calendar view — {selected_month.strftime('%B %Y')}")

    show_totals = st.toggle("Show total packs per day", value=True)

    tab_all, tab_w, tab_b, tab_m = st.tabs(["All", "Weekly", "Biweekly", "Monthly"])

    with tab_all:
        schedule = generate_month_schedule(selected_month, freq_filter=None)
        render_calendar_html(selected_month, schedule, "All deliveries", show_totals)

    with tab_w:
        schedule = generate_month_schedule(selected_month, freq_filter=1)
        render_calendar_html(selected_month, schedule, "Weekly deliveries only", show_totals)

    with tab_b:
        schedule = generate_month_schedule(selected_month, freq_filter=2)
        render_calendar_html(selected_month, schedule, "Biweekly deliveries only", show_totals)

    with tab_m:
        schedule = generate_month_schedule(selected_month, freq_filter=4)
        render_calendar_html(selected_month, schedule, "Monthly (4-week) deliveries only", show_totals)


# ============================================================
# UI: Print PDFs Tab
# ============================================================
def print_tab(selected_month: date):
    st.subheader("Print / Download PDF (Landscape, Single Page)")

    c1, c2, c3 = st.columns([1.4, 1.2, 1.2])
    with c1:
        filt = st.selectbox(
            "Which schedule?",
            options=["All", "Weekly", "Biweekly", "Monthly"],
            index=0,
        )
    with c2:
        show_totals = st.toggle("Show total packs per day (PDF)", value=True)
    with c3:
        st.caption("PDF is always 1 page, landscape, month-only cells.")

    freq_filter = None
    if filt == "Weekly":
        freq_filter = 1
    elif filt == "Biweekly":
        freq_filter = 2
    elif filt == "Monthly":
        freq_filter = 4

    pdf_bytes = build_month_pdf(selected_month, freq_filter=freq_filter, show_totals=show_totals)

    filename = f"blisterpack_{selected_month.strftime('%Y_%m')}_{filt.lower()}.pdf"
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        type="primary",
    )


# ============================================================
# MAIN
# ============================================================
ensure_db()

st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption("Auto-generates your month delivery sheet from patient frequency (Weekly / Biweekly / Monthly). Use Overrides for exceptions.")

# Lock
pin_gate()

# Month picker (plan ahead)
st.markdown("### Planning controls")
c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
with c1:
    view_month = st.date_input("Pick any date in the month you want to view/print", value=date.today())
with c2:
    # quick nav month +/-1
    if st.button("◀ Prev month"):
        d = date(view_month.year, view_month.month, 1) - timedelta(days=1)
        st.session_state["bp_month_jump"] = d
        st.rerun()
    if st.button("Next month ▶"):
        last = pycal.monthrange(view_month.year, view_month.month)[1]
        d = date(view_month.year, view_month.month, last) + timedelta(days=1)
        st.session_state["bp_month_jump"] = d
        st.rerun()
with c3:
    st.info("Tip: You can schedule now and print future months anytime. Overrides handle holidays/weekends/special requests.")

if "bp_month_jump" in st.session_state:
    view_month = st.session_state.pop("bp_month_jump")

st.divider()

# Tabs
tab_cal, tab_pat, tab_ovr, tab_pdf = st.tabs(["📅 Calendar (default)", "👥 Patients", "🧩 Overrides", "🖨️ Print PDFs"])

with tab_cal:
    calendar_tab(view_month)

with tab_pat:
    patients_tab()

with tab_ovr:
    overrides_tab(view_month)

with tab_pdf:
    print_tab(view_month)
