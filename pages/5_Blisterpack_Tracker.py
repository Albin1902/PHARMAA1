import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
import calendar
from io import BytesIO

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Blister Pack Tracker", layout="wide")
st.title("Blister Pack Calendar + Delivery Tracker")
st.caption("Use initials/ticket # instead of PHI if this app is public/shared.")

# -----------------------------
# SQLite setup (reuse same DB file as your other pages)
# -----------------------------
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "waiters.db")
os.makedirs(DATA_DIR, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT NOT NULL,            -- initials / ticket #
                rotation_type TEXT NOT NULL,           -- weekly | biweekly | monthly
                rotation_group TEXT NOT NULL,          -- R1 | R2 | R3
                next_due DATE NOT NULL,
                last_done DATE,
                notes TEXT,
                needs_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bp_next_due ON bp_clients(next_due)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bp_rot ON bp_clients(rotation_type, rotation_group)")
        conn.commit()

init_db()

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

# -----------------------------
# Date helpers
# -----------------------------
def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))

def advance_due(d: date, rotation_type: str) -> date:
    if rotation_type == "weekly":
        return d + timedelta(days=7)
    if rotation_type == "biweekly":
        return d + timedelta(days=14)
    return add_months(d, 1)

def days_until(d: date) -> int:
    return (d - date.today()).days

def start_of_week(d: date) -> date:
    # Monday start
    return d - timedelta(days=d.weekday())

# -----------------------------
# DB operations
# -----------------------------
def fetch_clients():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, patient_code, rotation_type, rotation_group, next_due, last_done,
                   notes, needs_order, created_at, updated_at
            FROM bp_clients
            ORDER BY next_due ASC
            """
        ).fetchall()
    out = []
    for r in rows:
        rr = dict(r)
        rr["next_due"] = date.fromisoformat(rr["next_due"])
        rr["last_done"] = date.fromisoformat(rr["last_done"]) if rr["last_done"] else None
        rr["needs_order"] = bool(rr["needs_order"])
        out.append(rr)
    return out

def add_client(patient_code: str, rotation_type: str, rotation_group: str, next_due: date, notes: str, needs_order: bool):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO bp_clients (patient_code, rotation_type, rotation_group, next_due, last_done, notes, needs_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (patient_code.strip(), rotation_type, rotation_group, next_due.isoformat(), notes.strip(), int(needs_order), now, now),
        )
        conn.commit()

def update_client(client_id: int, patient_code: str, rotation_type: str, rotation_group: str, next_due: date, notes: str, needs_order: bool):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE bp_clients
            SET patient_code=?, rotation_type=?, rotation_group=?, next_due=?, notes=?, needs_order=?, updated_at=?
            WHERE id=?
            """,
            (patient_code.strip(), rotation_type, rotation_group, next_due.isoformat(), notes.strip(), int(needs_order), now, client_id),
        )
        conn.commit()

def mark_delivered(client_id: int):
    today = date.today()
    with get_conn() as conn:
        row = conn.execute("SELECT next_due, rotation_type FROM bp_clients WHERE id=?", (client_id,)).fetchone()
        if not row:
            return
        next_due = date.fromisoformat(row["next_due"])
        rotation_type = row["rotation_type"]
        new_due = advance_due(next_due, rotation_type)

        conn.execute(
            """
            UPDATE bp_clients
            SET last_done=?, next_due=?, updated_at=?
            WHERE id=?
            """,
            (today.isoformat(), new_due.isoformat(), utc_iso(), client_id),
        )
        conn.commit()

def reschedule(client_id: int, new_due: date):
    with get_conn() as conn:
        conn.execute(
            "UPDATE bp_clients SET next_due=?, updated_at=? WHERE id=?",
            (new_due.isoformat(), utc_iso(), client_id),
        )
        conn.commit()

def delete_client(client_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM bp_clients WHERE id=?", (client_id,))
        conn.commit()

# -----------------------------
# Build maps for calendar display
# -----------------------------
def month_due_map(clients, year: int, month: int):
    mp = {}  # day -> list[str]
    for c in clients:
        d = c["next_due"]
        if d.year == year and d.month == month:
            tag = f"{c['patient_code']} ({c['rotation_group']})"
            mp.setdefault(d.day, []).append(tag)
    # sort entries per day
    for k in mp:
        mp[k] = sorted(mp[k])
    return mp

def week_due_map(clients, week_start: date):
    # week_start = Monday
    mp = {week_start + timedelta(days=i): [] for i in range(7)}
    for c in clients:
        d = c["next_due"]
        if week_start <= d <= (week_start + timedelta(days=6)):
            mp[d].append(f"{c['patient_code']} ({c['rotation_group']})")
    for k in mp:
        mp[k] = sorted(mp[k])
    return mp

# -----------------------------
# PDF generation
# -----------------------------
def build_month_pdf(year: int, month: int, clients) -> bytes:
    styles = getSampleStyleSheet()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), leftMargin=0.4*inch, rightMargin=0.4*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)

    title = f"Blister Pack Schedule — {calendar.month_name[month]} {year}"
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.15*inch)]
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    mp = month_due_map(clients, year, month)
    cal = calendar.monthcalendar(year, month)

    # header row
    data = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]

    for week in cal:
        row = []
        for day in week:
            if day == 0:
                row.append("")
            else:
                items = mp.get(day, [])
                # limit shown lines to keep printable; overflow indicates more
                max_lines = 5
                shown = items[:max_lines]
                extra = len(items) - len(shown)
                cell = f"{day}\n" + ("\n".join(shown) if shown else "")
                if extra > 0:
                    cell += f"\n(+{extra} more)"
                row.append(cell.strip())
        data.append(row)

    tbl = Table(data, colWidths=[(10.5*inch)/7]*7, rowHeights=[0.45*inch] + [1.1*inch]*len(cal))
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("LEFTPADDING", (0,1), (-1,-1), 6),
        ("RIGHTPADDING", (0,1), (-1,-1), 6),
        ("TOPPADDING", (0,1), (-1,-1), 6),
        ("BOTTOMPADDING", (0,1), (-1,-1), 6),
    ]))

    story.append(tbl)
    doc.build(story)
    return buf.getvalue()

def build_week_pdf(week_start: date, clients) -> bytes:
    styles = getSampleStyleSheet()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)

    week_end = week_start + timedelta(days=6)
    title = f"Blister Pack — Weekly Sheet ({week_start.isoformat()} to {week_end.isoformat()})"
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.15*inch)]
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    mp = week_due_map(clients, week_start)

    data = [["Day", "Due (Patient / Rotation)"]]
    for i in range(7):
        d = week_start + timedelta(days=i)
        day_name = d.strftime("%a %Y-%m-%d")
        items = mp[d]
        cell = "\n".join(items) if items else ""
        data.append([day_name, cell])

    tbl = Table(data, colWidths=[2.2*inch, 4.9*inch])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,1), (-1,-1), 10),
        ("LEFTPADDING", (0,1), (-1,-1), 6),
        ("RIGHTPADDING", (0,1), (-1,-1), 6),
        ("TOPPADDING", (0,1), (-1,-1), 6),
        ("BOTTOMPADDING", (0,1), (-1,-1), 6),
    ]))

    story.append(tbl)
    doc.build(story)
    return buf.getvalue()

# -----------------------------
# Sidebar: Add / Edit client
# -----------------------------
with st.sidebar:
    st.header("Add / Edit Blister Pack")
    mode = st.radio("Mode", ["Add new", "Edit existing"], horizontal=True)

    clients = fetch_clients()
    client_by_id = {c["id"]: c for c in clients}

    if mode == "Edit existing" and clients:
        pick = st.selectbox(
            "Select",
            options=[c["id"] for c in clients],
            format_func=lambda cid: f"{client_by_id[cid]['patient_code']} • {client_by_id[cid]['rotation_type']} {client_by_id[cid]['rotation_group']} (#{cid})"
        )
        cur = client_by_id[pick]
        patient_code = st.text_input("Patient code", value=cur["patient_code"])
        rotation_type = st.selectbox("Rotation type", ["weekly", "biweekly", "monthly"],
                                     index=["weekly","biweekly","monthly"].index(cur["rotation_type"]))
        rotation_group = st.selectbox("Rotation group", ["R1", "R2", "R3"],
                                      index=["R1","R2","R3"].index(cur["rotation_group"]))
        next_due = st.date_input("Next due date", value=cur["next_due"])
        needs_order = st.checkbox("Needs ordering", value=cur["needs_order"])
        notes = st.text_area("Notes", value=cur.get("notes","") or "", height=90)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save changes", type="primary", use_container_width=True):
                update_client(pick, patient_code, rotation_type, rotation_group, next_due, notes, needs_order)
                st.success("Updated.")
                st.rerun()
        with c2:
            if st.button("Mark delivered", use_container_width=True):
                mark_delivered(pick)
                st.success("Delivered + advanced next due.")
                st.rerun()
        with c3:
            confirm = st.checkbox("Confirm delete", key="bp_del_confirm")
            if st.button("Delete", disabled=not confirm, use_container_width=True):
                delete_client(pick)
                st.success("Deleted.")
                st.rerun()

        st.divider()
        st.caption("Reschedule quickly")
        new_due = st.date_input("Reschedule to", value=cur["next_due"], key="bp_resched")
        if st.button("Apply reschedule", use_container_width=True):
            reschedule(pick, new_due)
            st.success("Rescheduled.")
            st.rerun()

    else:
        patient_code = st.text_input("Patient code (initials/ticket #)", placeholder="e.g., AB-12")
        rotation_type = st.selectbox("Rotation type", ["weekly", "biweekly", "monthly"])
        rotation_group = st.selectbox("Rotation group", ["R1", "R2", "R3"])
        next_due = st.date_input("First/Next due date", value=date.today())
        needs_order = st.checkbox("Needs ordering", value=False)
        notes = st.text_area("Notes (optional)", height=90)

        if st.button("Add schedule", type="primary", use_container_width=True):
            if not patient_code.strip():
                st.error("Patient code is required.")
            else:
                add_client(patient_code, rotation_type, rotation_group, next_due, notes, needs_order)
                st.success("Added.")
                st.rerun()

# -----------------------------
# Main views
# -----------------------------
clients = fetch_clients()

tab_upcoming, tab_calendar, tab_print = st.tabs(["Upcoming", "Calendar", "Print PDFs"])

# Upcoming dataframe
rows = []
for c in clients:
    rows.append({
        "ID": c["id"],
        "Patient": c["patient_code"],
        "Rotation": f"{c['rotation_type']} {c['rotation_group']}",
        "Next due": c["next_due"].isoformat(),
        "Days until": days_until(c["next_due"]),
        "Needs order": "YES" if c["needs_order"] else "",
        "Last done": c["last_done"].isoformat() if c["last_done"] else "",
        "Notes": (c.get("notes","") or ""),
    })
df = pd.DataFrame(rows)

with tab_upcoming:
    st.subheader("Upcoming deliveries / pickups")
    if df.empty:
        st.info("No schedules yet. Add some from the sidebar.")
    else:
        horizon = st.selectbox("Show horizon", [7, 14, 30, 60], index=2)
        view = df.copy()
        view["Days until"] = view["Days until"].astype(int)
        view = view[view["Days until"] <= horizon].sort_values(["Days until", "Next due"])
        st.dataframe(view, use_container_width=True, hide_index=True)

with tab_calendar:
    st.subheader("Calendar view (NEXT due only)")
    if df.empty:
        st.info("No schedules yet.")
    else:
        today = date.today()
        year = today.year
        month = today.month

        mp = month_due_map(clients, year, month)
        cal = calendar.monthcalendar(year, month)

        grid = []
        for week in cal:
            row = []
            for day in week:
                if day == 0:
                    row.append("")
                else:
                    items = mp.get(day, [])
                    cell = str(day)
                    if items:
                        cell += "\n" + "\n".join(items[:6])
                        if len(items) > 6:
                            cell += f"\n(+{len(items)-6} more)"
                    row.append(cell)
            grid.append(row)

        cal_df = pd.DataFrame(grid, columns=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
        st.caption("After you mark delivered, the next due advances automatically.")

with tab_print:
    st.subheader("Print-ready PDFs")
    if not clients:
        st.info("Add schedules first.")
    else:
        today = date.today()
        year = today.year
        month = today.month
        wk_start = start_of_week(today)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Current Month")
            month_pdf = build_month_pdf(year, month, clients)
            st.download_button(
                label=f"Download {calendar.month_name[month]} {year} (PDF)",
                data=month_pdf,
                file_name=f"blisterpack_{year}_{month:02d}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Open the PDF → Print.")

        with col2:
            st.markdown("### Current Week")
            week_pdf = build_week_pdf(wk_start, clients)
            st.download_button(
                label=f"Download Week of {wk_start.isoformat()} (PDF)",
                data=week_pdf,
                file_name=f"blisterpack_week_{wk_start.isoformat()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Weekly sheet is a list by day (Mon–Sun).")
