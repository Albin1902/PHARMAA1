import os
import sqlite3
from datetime import datetime, timedelta, timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Waiter CRM", layout="wide")
st.title("Waiter CRM (Cash / Pickups)")
st.caption("Tip: Use initials or a ticket number. Don’t enter PHI in a public app.")

# Auto-refresh every 15 seconds so timers update without manual refresh
st_autorefresh(interval=15_000, key="waiter_refresh")

# -----------------------------
# SQLite setup (simple persistence)
# -----------------------------
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "waiters.db")

os.makedirs(DATA_DIR, exist_ok=True)

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

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
                status TEXT NOT NULL DEFAULT 'active',  -- active | done
                done_at TEXT
            )
            """
        )
        conn.commit()

init_db()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)

def add_waiter(patient: str, medication: str, minutes: int, note: str = ""):
    created = datetime.now(timezone.utc)
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
                (utc_now_iso(), row_id),
            )
        else:
            conn.execute(
                "UPDATE waiters SET status='active', done_at=NULL WHERE id=?",
                (row_id,),
            )
        conn.commit()

def extend_due(row_id: int, add_minutes: int):
    with get_conn() as conn:
        cur = conn.execute("SELECT due_at FROM waiters WHERE id=?", (row_id,))
        row = cur.fetchone()
        if not row:
            return
        due = parse_iso(row[0])
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
        rows = cur.fetchall()
    # convert to dicts
    cols = ["id", "created_at", "due_at", "patient", "medication", "note", "status", "done_at"]
    return [dict(zip(cols, r)) for r in rows]

def fmt_dt_local(iso_str: str) -> str:
    # Show in local browser time-ish (still server time), but readable.
    dt = parse_iso(iso_str)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

def remaining_text(due_at_iso: str) -> tuple[str, bool]:
    now = datetime.now(timezone.utc)
    due = parse_iso(due_at_iso)
    delta = due - now
    secs = int(delta.total_seconds())
    overdue = secs < 0
    secs_abs = abs(secs)
    mins = secs_abs // 60
    s = secs_abs % 60
    if overdue:
        return f"OVERDUE by {mins}m {s:02d}s", True
    return f"{mins}m {s:02d}s left", False


# -----------------------------
# Add new waiter
# -----------------------------
st.subheader("Add a waiter")

c1, c2, c3, c4 = st.columns([1, 1.3, 0.8, 1.2])

with c1:
    patient = st.text_input("Patient (initials / ticket #)", placeholder="e.g., AB / #42")
with c2:
    medication = st.text_input("Medication / Request", placeholder="e.g., Amox 250/5, refill, prior auth")
with c3:
    minutes = st.number_input("Timer (minutes)", min_value=1, value=15, step=1)
with c4:
    note = st.text_input("Note (optional)", placeholder="e.g., call prescriber, waiting on adjudication")

add_btn = st.button("Add waiter", type="primary", use_container_width=True)
if add_btn:
    if not patient.strip() or not medication.strip():
        st.error("Need at least Patient + Medication.")
    else:
        add_waiter(patient, medication, int(minutes), note)
        st.success("Added.")
        st.rerun()

st.divider()

# -----------------------------
# View / Filters
# -----------------------------
rows = fetch_all()

active = [r for r in rows if r["status"] == "active"]
done = [r for r in rows if r["status"] == "done"]

overdue = []
not_overdue = []
for r in active:
    _, is_overdue = remaining_text(r["due_at"])
    (overdue if is_overdue else not_overdue).append(r)

tab_active, tab_overdue, tab_done = st.tabs(
    [f"Active ({len(active)})", f"Overdue ({len(overdue)})", f"Done / Archive ({len(done)})"]
)

def render_list(items, mode: str):
    if not items:
        st.info("Nothing here.")
        return

    for r in items:
        rem, is_overdue = remaining_text(r["due_at"])
        header = f"{r['patient']} — {r['medication']}"

        # Card-like container
        with st.container(border=True):
            top = st.columns([2.2, 1.2, 1.2, 1.4, 1.2])

            with top[0]:
                if is_overdue and mode != "done":
                    st.markdown(f"### 🔴 {header}")
                else:
                    st.markdown(f"### {header}")
                if r.get("note"):
                    st.caption(r["note"])

            with top[1]:
                st.write("**Created**")
                st.write(fmt_dt_local(r["created_at"]))

            with top[2]:
                st.write("**Due**")
                st.write(fmt_dt_local(r["due_at"]))

            with top[3]:
                st.write("**Status**")
                if mode == "done":
                    st.success("DONE")
                    if r.get("done_at"):
                        st.caption(f"Done at: {fmt_dt_local(r['done_at'])}")
                else:
                    if is_overdue:
                        st.error(rem)
                    else:
                        st.info(rem)

            with top[4]:
                # Action buttons
                if mode == "done":
                    if st.button("Undo", key=f"undo_{r['id']}"):
                        set_done(r["id"], False)
                        st.rerun()
                    if st.button("Delete", key=f"del_done_{r['id']}"):
                        delete_row(r["id"])
                        st.rerun()
                else:
                    if st.button("Mark done", key=f"done_{r['id']}"):
                        set_done(r["id"], True)
                        st.rerun()

                    snooze = st.selectbox(
                        "Extend",
                        options=[5, 10, 15, 30],
                        index=0,
                        key=f"snooze_{r['id']}",
                        label_visibility="collapsed",
                    )
                    if st.button(f"+{snooze}m", key=f"ext_{r['id']}"):
                        extend_due(r["id"], int(snooze))
                        st.rerun()

                    if st.button("Delete", key=f"del_{r['id']}"):
                        delete_row(r["id"])
                        st.rerun()


with tab_active:
    st.subheader("Active waiters (sorted by due time)")
    render_list(active, mode="active")

with tab_overdue:
    st.subheader("Overdue waiters")
    render_list(overdue, mode="overdue")

with tab_done:
    st.subheader("Archive (done)")
    render_list(done, mode="done")

st.caption("Note: SQLite persists across refreshes. On Streamlit Cloud, redeploys or server resets can clear local files—if you need guaranteed persistence, use a hosted DB (Supabase/Postgres).")
