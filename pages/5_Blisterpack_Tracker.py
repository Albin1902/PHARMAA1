import os
import sqlite3
import calendar
from datetime import date, datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------
# Page config + light CSS (landscape feel)
# -----------------------------
st.set_page_config(page_title="Blisterpack Tracker", layout="wide")
st.markdown(
    """
    <style>
    /* reduce vertical wasted space a bit */
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    /* make dataframes look tighter */
    [data-testid="stDataFrame"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Blister Pack Schedule + Delivery Tracker")
st.caption("Use patient codes (initials/ticket#). Do NOT enter PHI if this app is public.")

# -----------------------------
# DB setup
# -----------------------------
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "waiters.db")  # reuse your existing db file
os.makedirs(DATA_DIR, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT NOT NULL,
                cadence TEXT NOT NULL,         -- weekly | biweekly | monthly
                rotation TEXT NOT NULL,        -- R1 | R2 | R3 (your route/rotation)
                day_of_week INTEGER NOT NULL,  -- 0=Mon ... 6=Sun
                week_slot TEXT NOT NULL,       -- W1/W2/W3/W4 or 'NA'
                pkgs INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_delivery_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schedule_id INTEGER NOT NULL,
                due_date TEXT NOT NULL,          -- YYYY-MM-DD
                status TEXT NOT NULL,            -- pending | done
                note TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(schedule_id, due_date)
            )
            """
        )
        # settings table for cycle anchor (fixes W1–W4 logic)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bp_sched_active ON bp_schedule(active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bp_sched_day ON bp_schedule(day_of_week)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bp_log_due ON bp_delivery_log(due_date)")
        conn.commit()

init_db()

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKSLOTS = ["W1","W2","W3","W4"]
ROTATIONS = ["R1","R2","R3"]
CADENCES = ["weekly","biweekly","monthly"]

# -----------------------------
# Date helpers
# -----------------------------
def start_of_week(d):
    return d - timedelta(days=d.weekday())  # Monday

def month_range(d):
    first = date(d.year, d.month, 1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = date(d.year, d.month, last_day)
    return first, last

# -----------------------------
# Settings helpers (cycle anchor)
# -----------------------------
def get_setting(key, default=None):
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM bp_settings WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    return row["value"]

def set_setting(key, value):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bp_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, str(value)),
        )
        conn.commit()

def get_anchor_monday():
    # stored as YYYY-MM-DD (must be a Monday)
    raw = get_setting("cycle_anchor_monday", None)
    if raw:
        try:
            return date.fromisoformat(raw)
        except Exception:
            pass
    # default: current week Monday (works immediately, but you SHOULD set it)
    d = start_of_week(date.today())
    set_setting("cycle_anchor_monday", d.isoformat())
    return d

def cycle_week_slot(week_start, anchor_monday):
    # anchor_monday MUST be a Monday
    weeks = (week_start - anchor_monday).days // 7
    idx = (weeks % 4) + 1
    return f"W{idx}"

# -----------------------------
# DB ops
# -----------------------------
def fetch_schedule(active_only=True):
    with get_conn() as conn:
        if active_only:
            rows = conn.execute(
                """
                SELECT * FROM bp_schedule
                WHERE active=1
                ORDER BY cadence, week_slot, day_of_week, rotation, patient_code
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM bp_schedule
                ORDER BY active DESC, cadence, week_slot, day_of_week, rotation, patient_code
                """
            ).fetchall()
    return [dict(r) for r in rows]

def add_schedule(patient_code, cadence, rotation, day_of_week, week_slot, pkgs, notes):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bp_schedule
            (patient_code, cadence, rotation, day_of_week, week_slot, pkgs, notes, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (patient_code.strip(), cadence, rotation, int(day_of_week), week_slot, int(pkgs), (notes or "").strip(), now, now),
        )
        conn.commit()

def update_schedule(row_id, patient_code, cadence, rotation, day_of_week, week_slot, pkgs, notes, active):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE bp_schedule
            SET patient_code=?, cadence=?, rotation=?, day_of_week=?, week_slot=?, pkgs=?, notes=?, active=?, updated_at=?
            WHERE id=?
            """,
            (patient_code.strip(), cadence, rotation, int(day_of_week), week_slot, int(pkgs), (notes or "").strip(), int(active), now, int(row_id)),
        )
        conn.commit()

def delete_schedule(row_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM bp_schedule WHERE id=?", (int(row_id),))
        conn.execute("DELETE FROM bp_delivery_log WHERE schedule_id=?", (int(row_id),))
        conn.commit()

def get_log(schedule_id, due_date):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, note FROM bp_delivery_log WHERE schedule_id=? AND due_date=?",
            (int(schedule_id), due_date.isoformat())
        ).fetchone()
    if not row:
        return {"status": "pending", "note": ""}
    return {"status": row["status"], "note": row["note"] or ""}

def set_log(schedule_id, due_date, status, note=""):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bp_delivery_log (schedule_id, due_date, status, note, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(schedule_id, due_date)
            DO UPDATE SET status=excluded.status, note=excluded.note, updated_at=excluded.updated_at
            """,
            (int(schedule_id), due_date.isoformat(), status, (note or "").strip(), now),
        )
        conn.commit()

# -----------------------------
# Occurrence builder (THIS WEEK)
# -----------------------------
def build_week_occurrences(week_start, schedule_rows, slot):
    occ = []
    for r in schedule_rows:
        dow = int(r["day_of_week"])
        due = week_start + timedelta(days=dow)

        if r["cadence"] == "weekly":
            include = True
        else:
            include = (r["week_slot"] == slot)

        if include:
            occ.append({
                "schedule_id": r["id"],
                "due_date": due,
                "day": DAYS[dow],
                "patient_code": r["patient_code"],
                "rotation": r["rotation"],
                "cadence": r["cadence"],
                "pkgs": int(r["pkgs"]),
                "sched_note": (r.get("notes","") or "")
            })

    occ.sort(key=lambda x: (x["due_date"], x["rotation"], x["patient_code"]))
    return occ

# -----------------------------
# PDF builders
# -----------------------------
def build_week_pdf_landscape(week_start, occs, slot, anchor_monday):
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=10,
        spaceAfter=0,
        spaceBefore=0,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),  # ✅ landscape weekly
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
    )

    week_end = week_start + timedelta(days=6)
    title = f"Blister Pack — Weekly Sheet ({week_start.isoformat()} to {week_end.isoformat()})"
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.10*inch)]
    story.append(Paragraph(f"Cycle slot: <b>{slot}</b>  |  Anchor Monday: {anchor_monday.isoformat()}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.15*inch))

    by_day = {}
    for o in occs:
        by_day.setdefault(o["due_date"], []).append(o)

    data = [[
        Paragraph("<b>Day</b>", cell_style),
        Paragraph("<b>Patient / Rotation</b>", cell_style),
        Paragraph("<b>Cadence</b>", cell_style),
        Paragraph("<b>Pkgs</b>", cell_style),
        Paragraph("<b>D/P/U</b>", cell_style),
        Paragraph("<b>Notes</b>", cell_style),
    ]]

    for i in range(7):
        d = week_start + timedelta(days=i)
        day_label = d.strftime("%a %Y-%m-%d")
        items = by_day.get(d, [])

        if not items:
            data.append([Paragraph(day_label, cell_style), "", "", "", Paragraph("☐", cell_style), ""])
            continue

        for idx, o in enumerate(items):
            label = day_label if idx == 0 else ""
            patient_txt = f"{o['patient_code']} ({o['rotation']})"
            cadence_txt = o["cadence"]
            pkgs_txt = str(o["pkgs"]) if o["pkgs"] else ""
            note_txt = o["sched_note"] or ""
            data.append([
                Paragraph(label, cell_style),
                Paragraph(patient_txt, cell_style),
                Paragraph(cadence_txt, cell_style),
                Paragraph(pkgs_txt, cell_style),
                Paragraph("☐", cell_style),
                Paragraph(note_txt, cell_style),
            ])

    # Wider page, so we can make it readable.
    tbl = Table(
        data,
        colWidths=[1.55*inch, 3.2*inch, 1.1*inch, 0.7*inch, 0.7*inch, 4.2*inch],
    )
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))

    story.append(tbl)
    doc.build(story)
    return buf.getvalue()

def build_month_calendar_pdf(year, month, occs_for_month):
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        spaceAfter=0,
        spaceBefore=0,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=0.35*inch,
        rightMargin=0.35*inch,
        topMargin=0.35*inch,
        bottomMargin=0.35*inch,
    )

    title = f"Blister Pack — Calendar ({calendar.month_name[month]} {year})"
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.12*inch)]
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.15*inch))

    cal = calendar.monthcalendar(year, month)

    def make_cell(day):
        if day == 0:
            return Paragraph("", cell_style), 1

        items = occs_for_month.get(day, [])
        max_names = 10  # ✅ accommodate ~10 per day
        shown = items[:max_names]
        extra = len(items) - len(shown)

        lines = ["<b>%d</b>" % day]
        for x in shown:
            x = x.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            lines.append(x)
        if extra > 0:
            lines.append("(+%d more)" % extra)

        html = "<br/>".join(lines)
        line_count = 1 + len(shown) + (1 if extra > 0 else 0)
        return Paragraph(html, cell_style), line_count

    data = [[Paragraph(x, styles["Heading4"]) for x in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]]]

    row_heights = [0.35*inch]
    base_padding = 0.18*inch

    for week in cal:
        row = []
        week_max_lines = 1
        for day in week:
            cell, nlines = make_cell(day)
            row.append(cell)
            week_max_lines = max(week_max_lines, nlines)
        data.append(row)
        row_heights.append(base_padding + week_max_lines * 0.14 * inch)

    tbl = Table(data, colWidths=[(10.7*inch)/7]*7, rowHeights=row_heights)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,1), (-1,-1), 5),
        ("RIGHTPADDING", (0,1), (-1,-1), 5),
        ("TOPPADDING", (0,1), (-1,-1), 4),
        ("BOTTOMPADDING", (0,1), (-1,-1), 4),
    ]))

    story.append(tbl)
    doc.build(story)
    return buf.getvalue()

# -----------------------------
# Sidebar: cycle settings + schedule CRUD
# -----------------------------
with st.sidebar:
    st.header("Blisterpack Tracker")

    st.subheader("Cycle settings (fixes W1–W4)")
    anchor = get_anchor_monday()
    picked = st.date_input("Cycle anchor Monday (W1 starts here)", value=anchor)

    if picked.weekday() != 0:
        st.error("Pick a MONDAY. This controls W1–W4 across months.")
    else:
        if picked != anchor:
            set_setting("cycle_anchor_monday", picked.isoformat())
            st.success("Anchor saved. Reloading…")
            st.rerun()

    st.caption("Tip: Set this once to match your pharmacy’s 4-week blister cycle.")

    st.divider()
    st.subheader("Add / Edit BP Schedule")

    mode = st.radio("Mode", ["Add", "Edit"], horizontal=True)
    all_rows = fetch_schedule(active_only=False)

    if mode == "Add":
        patient_code = st.text_input("Patient code (NO names)", placeholder="e.g., AB-42")
        cadence = st.selectbox("Cadence", CADENCES, index=0)
        rotation = st.selectbox("Rotation", ROTATIONS, index=0)
        day_of_week = st.selectbox("Day", list(range(7)), format_func=lambda i: DAYS[i], index=1)

        if cadence == "weekly":
            week_slot = "NA"
            st.info("Weekly ignores W1–W4.")
        else:
            week_slot = st.selectbox("Week slot (W1–W4)", WEEKSLOTS, index=2)

        pkgs = st.number_input("Pkgs (optional)", min_value=0, value=0, step=1)
        notes = st.text_area("Notes (optional)", height=80)

        if st.button("Add schedule", type="primary", use_container_width=True):
            if not patient_code.strip():
                st.error("Patient code required.")
            else:
                add_schedule(patient_code, cadence, rotation, day_of_week, week_slot, pkgs, notes)
                st.success("Added.")
                st.rerun()

    else:
        if not all_rows:
            st.info("No schedules yet.")
        else:
            def fmt(r):
                return f"{r['patient_code']} • {r['cadence']} • {r['rotation']} • {DAYS[int(r['day_of_week'])]} • {r['week_slot']} (#{r['id']})"

            pick_id = st.selectbox(
                "Select schedule",
                options=[r["id"] for r in all_rows],
                format_func=lambda rid: fmt(next(x for x in all_rows if x["id"] == rid))
            )
            cur = next(r for r in all_rows if r["id"] == pick_id)

            patient_code = st.text_input("Patient code", value=cur["patient_code"])
            cadence = st.selectbox("Cadence", CADENCES, index=CADENCES.index(cur["cadence"]))
            rotation = st.selectbox("Rotation", ROTATIONS, index=ROTATIONS.index(cur["rotation"]))
            day_of_week = st.selectbox("Day", list(range(7)), format_func=lambda i: DAYS[i], index=int(cur["day_of_week"]))

            if cadence == "weekly":
                week_slot = "NA"
                st.info("Weekly ignores W1–W4.")
            else:
                ws = cur["week_slot"] if cur["week_slot"] in WEEKSLOTS else "W1"
                week_slot = st.selectbox("Week slot (W1–W4)", WEEKSLOTS, index=WEEKSLOTS.index(ws))

            pkgs = st.number_input("Pkgs", min_value=0, value=int(cur["pkgs"]), step=1)
            notes = st.text_area("Notes", value=cur.get("notes","") or "", height=80)
            active = st.checkbox("Active", value=bool(cur["active"]))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save changes", type="primary", use_container_width=True):
                    update_schedule(pick_id, patient_code, cadence, rotation, day_of_week, week_slot, pkgs, notes, active)
                    st.success("Saved.")
                    st.rerun()
            with c2:
                confirm = st.checkbox("Confirm delete", key="bp_confirm_delete")
                if st.button("Delete", disabled=not confirm, use_container_width=True):
                    delete_schedule(pick_id)
                    st.success("Deleted.")
                    st.rerun()

# -----------------------------
# Main tabs (simpler)
# -----------------------------
schedule_active = fetch_schedule(active_only=True)

today = date.today()
week_start = start_of_week(today)
week_end = week_start + timedelta(days=6)

anchor_monday = get_anchor_monday()
slot = cycle_week_slot(week_start, anchor_monday)

tab_week, tab_month, tab_print = st.tabs(["This Week", "Month View", "Print PDFs"])

# -----------------------------
# This Week: simple board + one editable table
# -----------------------------
with tab_week:
    st.subheader(f"Current Week: {week_start.isoformat()} → {week_end.isoformat()}  |  Slot: {slot}")

    occs = build_week_occurrences(week_start, schedule_active, slot)

    if not occs:
        st.info("No schedules match this week. Add schedules from the sidebar.")
    else:
        # --- Board view (landscape-like)
        st.markdown("### Board view (simple)")
        by_day_text = {d: [] for d in range(7)}
        for o in occs:
            dow = o["due_date"].weekday()
            label = f"{o['patient_code']} ({o['rotation']}) • {o['cadence']}"
            if o["pkgs"]:
                label += f" • pkgs:{o['pkgs']}"
            if o["sched_note"]:
                label += f" • {o['sched_note']}"
            by_day_text[dow].append(label)

        board_row = {}
        for i in range(7):
            head = f"{DAYS[i][:3]} { (week_start + timedelta(days=i)).strftime('%m/%d') }"
            items = by_day_text[i]
            board_row[head] = "\n".join(items) if items else ""

        board_df = pd.DataFrame([board_row])
        st.dataframe(board_df, use_container_width=True, height=170)

        st.divider()

        # --- Tracker table (one place to edit done/notes)
        st.markdown("### Week tracker (edit + save once)")

        rows = []
        for o in occs:
            log = get_log(o["schedule_id"], o["due_date"])
            rows.append({
                "due_date": o["due_date"].isoformat(),
                "day": o["day"],
                "patient": o["patient_code"],
                "rotation": o["rotation"],
                "cadence": o["cadence"],
                "pkgs": o["pkgs"],
                "done": (log["status"] == "done"),
                "note": log["note"],
                "_schedule_id": o["schedule_id"],  # hidden helper
            })

        df = pd.DataFrame(rows)

        with st.form("week_save_form"):
            edited = st.data_editor(
                df.drop(columns=["_schedule_id"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "done": st.column_config.CheckboxColumn("Done"),
                    "note": st.column_config.TextColumn("Note", width="large"),
                },
                disabled=["due_date","day","patient","rotation","cadence","pkgs"],
            )
            save = st.form_submit_button("Save week updates", type="primary", use_container_width=True)

        if save:
            # compare edited 'done' + 'note' and persist
            edited_df = edited.copy()
            # re-attach schedule_id by merging on due_date + patient + rotation (stable enough)
            merged = pd.merge(
                df,
                edited_df,
                on=["due_date","day","patient","rotation","cadence","pkgs"],
                suffixes=("_old",""),
                how="left"
            )
            for _, r in merged.iterrows():
                due = date.fromisoformat(r["due_date"])
                status = "done" if bool(r["done"]) else "pending"
                note = str(r["note"]) if r["note"] is not None else ""
                set_log(int(r["_schedule_id"]), due, status, note)
            st.success("Saved.")
            st.rerun()

# -----------------------------
# Month view
# -----------------------------
with tab_month:
    st.subheader("Current Month (Calendar view)")
    y, m = today.year, today.month
    first, last = month_range(today)

    # build occurrences for each week covering the month using the SAME cycle slot logic
    occs_by_day = {}
    cur = start_of_week(first)
    while cur <= last:
        week_slot = cycle_week_slot(cur, anchor_monday)
        week_occs = build_week_occurrences(cur, schedule_active, week_slot)
        for o in week_occs:
            if o["due_date"].year == y and o["due_date"].month == m:
                occs_by_day.setdefault(o["due_date"].day, []).append(f"{o['patient_code']} ({o['rotation']})")
        cur = cur + timedelta(days=7)

    for k in occs_by_day:
        occs_by_day[k] = sorted(occs_by_day[k])

    cal = calendar.monthcalendar(y, m)
    grid = []
    for week in cal:
        row = []
        for daynum in week:
            if daynum == 0:
                row.append("")
            else:
                items = occs_by_day.get(daynum, [])
                cell = str(daynum)
                if items:
                    cell += "\n" + "\n".join(items[:10])
                    if len(items) > 10:
                        cell += f"\n(+{len(items)-10} more)"
                row.append(cell)
        grid.append(row)

    cal_df = pd.DataFrame(grid, columns=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    st.dataframe(cal_df, use_container_width=True, hide_index=True)
    st.caption("This view uses your cycle anchor so W1–W4 stays correct across months.")

# -----------------------------
# Print PDFs
# -----------------------------
with tab_print:
    st.subheader("Print PDFs (Landscape)")

    if not schedule_active:
        st.info("Add schedules first.")
    else:
        occs_week = build_week_occurrences(week_start, schedule_active, slot)

        y, m = today.year, today.month
        first, last = month_range(today)

        occs_by_day = {}
        cur = start_of_week(first)
        while cur <= last:
            week_slot = cycle_week_slot(cur, anchor_monday)
            week_occs = build_week_occurrences(cur, schedule_active, week_slot)
            for o in week_occs:
                if o["due_date"].year == y and o["due_date"].month == m:
                    occs_by_day.setdefault(o["due_date"].day, []).append(f"{o['patient_code']} ({o['rotation']})")
            cur = cur + timedelta(days=7)

        for k in occs_by_day:
            occs_by_day[k] = sorted(occs_by_day[k])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Weekly Sheet (Landscape)")
            week_pdf = build_week_pdf_landscape(week_start, occs_week, slot, anchor_monday)
            st.download_button(
                "Download Weekly PDF",
                data=week_pdf,
                file_name=f"bp_week_{week_start.isoformat()}_{slot}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col2:
            st.markdown("### Month Calendar (Landscape)")
            month_pdf = build_month_calendar_pdf(y, m, occs_by_day)
            st.download_button(
                "Download Month PDF",
                data=month_pdf,
                file_name=f"bp_month_{y}_{m:02d}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.caption("Print tip: Open PDF → Print → choose 'Fit to page' if needed.")
