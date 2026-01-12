# pages/3_Waiter_CRM.py

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Waiter CRM", layout="wide")
st.title("Waiter CRM (Cash / Pickups)")
st.caption("Use initials/ticket #. Don’t enter PHI in a public app.")

AUTO_REFRESH_SECONDS = 15
KEEP_DAYS = 2  # keep max 2 days
st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="waiter_refresh")

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "waiters.db")
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS waiters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                due_at TEXT NOT NULL,
                patient TEXT NOT NULL,
                medication TEXT NOT NULL,
                note TEXT,
                status TEXT NOT NULL DEFAULT 'active', -- active|done
                done_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waiters_status_due ON waiters(status, due_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waiters_created ON waiters(created_at)")
        conn.commit()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def fmt_local(iso_str: str) -> str:
    return parse_iso(iso_str).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def purge_old(days: int = KEEP_DAYS):
    cutoff = utcnow() - timedelta(days=days)
    with get_conn() as conn:
        conn.execute("DELETE FROM waiters WHERE created_at < ?", (cutoff.isoformat(),))
        conn.commit()


def add_waiter(patient: str, medication: str, minutes: int, note: str):
    created = utcnow()
    due = created + timedelta(minutes=int(minutes))
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO waiters (created_at, due_at, patient, medication, note, status, done_at)
            VALUES (?, ?, ?, ?, ?, 'active', NULL)
            """,
            (created.isoformat(), due.isoformat(), patient.strip(), medication.strip(), note.strip()),
        )
        conn.commit()


def set_done(row_id: int, done: bool):
    with get_conn() as conn:
        if done:
            conn.execute(
                "UPDATE waiters SET status='done', done_at=? WHERE id=?",
                (utcnow().isoformat(), row_id),
            )
        else:
            conn.execute(
                "UPDATE waiters SET status='active', done_at=NULL WHERE id=?",
                (row_id,),
            )
        conn.commit()


def extend_due(row_id: int, add_minutes: int):
    with get_conn() as conn:
        row = conn.execute("SELECT due_at FROM waiters WHERE id=?", (row_id,)).fetchone()
        if not row:
            return
        due = parse_iso(row["due_at"])
        new_due = due + timedelta(minutes=int(add_minutes))
        conn.execute("UPDATE waiters SET due_at=? WHERE id=?", (new_due.isoformat(), row_id))
        conn.commit()


def delete_row(row_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM waiters WHERE id=?", (row_id,))
        conn.commit()


def fetch_all():
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, created_at, due_at, patient, medication, note, status, done_at
            FROM waiters
            ORDER BY status ASC, due_at ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]


def remaining(due_iso: str) -> tuple[str, bool, int]:
    now = utcnow()
    due = parse_iso(due_iso)
    secs = int((due - now).total_seconds())
    overdue = secs < 0
    s = abs(secs)
    mins = s // 60
    sec = s % 60
    if overdue:
        return f"OVERDUE by {mins}m {sec:02d}s", True, secs
    return f"{mins}m {sec:02d}s left", False, secs


def to_minutes(val: int, unit: str) -> int:
    if unit == "minutes":
        return int(val)
    if unit == "hours":
        return int(val) * 60
    return int(val) * 1440


# -----------------------------
# Startup cleanup
# -----------------------------
init_db()
purge_old(KEEP_DAYS)

# -----------------------------
# Add waiter UI
# -----------------------------
st.subheader("Add a waiter")
c1, c2, c3, c4, c5 = st.columns([1.1, 1.8, 0.8, 0.9, 1.6])

with c1:
    patient = st.text_input("Patient (initials / ticket #)", placeholder="e.g., AB / #42")
with c2:
    medication = st.text_input("Medication / Request", placeholder="e.g., Amox 250/5, refill, prior auth")
with c3:
    tval = st.number_input("Timer", min_value=1, value=15, step=1)
with c4:
    tunit = st.selectbox("Unit", ["minutes", "hours", "days"], index=0)
with c5:
    note = st.text_input("Note (optional)", placeholder="e.g., waiting on adjudication")

if st.button("Add waiter", type="primary", use_container_width=True):
    if not patient.strip() or not medication.strip():
        st.error("Need at least Patient + Medication.")
    else:
        add_waiter(patient, medication, to_minutes(int(tval), tunit), note)
        st.success("Added.")
        st.rerun()

st.divider()

# -----------------------------
# View
# -----------------------------
rows = fetch_all()

search = st.text_input("Search (patient / med / note)", value="").strip().lower()
if search:
    def ok(r):
        blob = f"{r.get('patient','')} {r.get('medication','')} {r.get('note','')}".lower()
        return search in blob
    rows = [r for r in rows if ok(r)]

# Build a dataframe for clean display
table_rows = []
for r in rows:
    rem_txt, is_over, rem_secs = remaining(r["due_at"])
    table_rows.append({
        "id": r["id"],
        "patient": r["patient"],
        "medication": r["medication"],
        "note": r.get("note", ""),
        "created": fmt_local(r["created_at"]),
        "due": fmt_local(r["due_at"]),
        "remaining": rem_txt,
        "overdue": is_over,
        "status": r["status"],
    })

df = pd.DataFrame(table_rows)

active_df = df[df["status"] == "active"].copy()
overdue_df = active_df[active_df["overdue"] == True].copy()
done_df = df[df["status"] == "done"].copy()

tabA, tabO, tabD = st.tabs([f"Active ({len(active_df)})", f"Overdue ({len(overdue_df)})", f"Done ({len(done_df)})"])


def render_tab(dfx: pd.DataFrame, mode: str):
    if dfx.empty:
        st.info("Nothing here.")
        return

    show_cols = ["id", "patient", "medication", "note", "created", "due", "remaining"]
    st.dataframe(dfx[show_cols], use_container_width=True, hide_index=True)

    ids = dfx["id"].tolist()
    sel = st.selectbox("Select an entry to take action", options=ids, format_func=lambda x: f"#{x}", key=f"sel_{mode}")

    a1, a2, a3, a4 = st.columns([1, 1, 1, 2])

    with a1:
        if mode != "done":
            if st.button("Mark done", use_container_width=True, key=f"done_{mode}"):
                set_done(int(sel), True)
                st.rerun()
        else:
            if st.button("Undo", use_container_width=True, key=f"undo_{mode}"):
                set_done(int(sel), False)
                st.rerun()

    with a2:
        ext = st.selectbox("Extend", options=[5, 10, 15, 30, 60, 120], index=0, key=f"extsel_{mode}")
        if st.button(f"+{ext}m", use_container_width=True, key=f"extbtn_{mode}"):
            extend_due(int(sel), int(ext))
            st.rerun()

    with a3:
        confirm = st.checkbox("Confirm delete", key=f"conf_{mode}")
        if st.button("Delete", use_container_width=True, disabled=not confirm, key=f"del_{mode}"):
            delete_row(int(sel))
            st.rerun()

    with a4:
        st.caption(f"Auto-delete: entries older than {KEEP_DAYS} day(s) are removed on load.")


with tabA:
    st.subheader("Active")
    render_tab(active_df, "active")

with tabO:
    st.subheader("Overdue")
    render_tab(overdue_df, "overdue")

with tabD:
    st.subheader("Done / Archive")
    render_tab(done_df, "done")
