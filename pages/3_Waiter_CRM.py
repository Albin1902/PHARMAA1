# pages/3_Waiter_CRM.py

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Waiter CRM", layout="wide")
st.title("Waiter CRM (Cash / Pickups)")
st.caption("Use initials or ticket #. Don’t enter PHI in a public app.")

AUTO_REFRESH_SECONDS = 15
KEEP_DAYS = 2  # auto-delete entries older than this
st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="waiter_refresh")

# -----------------------------
# SQLite setup
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
            CREATE TABLE IF NOT EXISTS waiters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                due_at TEXT NOT NULL,
                patient TEXT NOT NULL,
                medication TEXT NOT NULL,
                note TEXT,
                status TEXT NOT NULL DEFAULT 'active',   -- active | done
                done_at TEXT
            )
            """
        )
        # helpful indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waiters_status_due ON waiters(status, due_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waiters_created ON waiters(created_at)")
        conn.commit()


init_db()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def fmt_dt_local(iso_str: str) -> str:
    dt = parse_iso(iso_str)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def remaining_text(due_at_iso: str) -> tuple[str, bool]:
    now = utc_now()
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


def timer_to_minutes(value: int, unit: str) -> int:
    if unit == "minutes":
        return int(value)
    if unit == "hours":
        return int(value) * 60
    return int(value) * 1440


def purge_old(days: int = KEEP_DAYS):
    """Delete anything (active or done) older than N days."""
    cutoff = utc_now() - timedelta(days=days)
    with get_conn() as conn:
        conn.execute("DELETE FROM waiters WHERE created_at < ?", (cutoff.isoformat(),))
        conn.commit()


def add_waiter(patient: str, medication: str, minutes: int, note: str = ""):
    created = utc_now()
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
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# Purge on every load (keeps DB clean for 2 days max)
purge_old(KEEP_DAYS)

# -----------------------------
# Controls (Top bar)
# -----------------------------
topbar = st.columns([1.2, 1.6, 1.2, 2.0])

with topbar[0]:
    search = st.text_input("Search (patient/med/note)", value="", placeholder="e.g., AB, amox, prior auth").strip().lower()

with topbar[1]:
    view_mode = st.radio("Show", ["All", "Only Active", "Only Overdue", "Only Done"], horizontal=True)

with topbar[2]:
    keep_days_ui = st.selectbox("Auto-delete after", options=[1, 2, 3, 7], index=1)
    # if you change it, purge immediately
    if keep_days_ui != KEEP_DAYS:
        KEEP_DAYS = keep_days_ui
        purge_old(KEEP_DAYS)

with topbar[3]:
    st.write("")
    st.caption(f"Auto-refresh: every {AUTO_REFRESH_SECONDS}s • Auto-delete: {KEEP_DAYS} day(s)")


st.divider()

# -----------------------------
# Add new waiter
# -----------------------------
st.subheader("Add a waiter")

c1, c2, c3, c4, c5 = st.columns([1.1, 1.6, 0.9, 0.9, 1.5])

with c1:
    patient = st.text_input("Patient (initials / ticket #)", placeholder="e.g., AB / #42")
with c2:
    medication = st.text_input("Medication / Request", placeholder="e.g., Amox 250/5, refill, prior auth")
with c3:
    timer_value = st.number_input("Timer", min_value=1, value=15, step=1)
with c4:
    timer_unit = st.selectbox("Unit", ["minutes", "hours", "days"], index=0)
with c5:
    note = st.text_input("Note (optional)", placeholder="e.g., waiting on adjudication")

minutes = timer_to_minutes(int(timer_value), timer_unit)

btns = st.columns([1, 1, 6])
with btns[0]:
    add_btn = st.button("Add", type="primary", use_container_width=True)
with btns[1]:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

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

# Apply search filter
if search:
    def matches(r):
        blob = f"{r.get('patient','')} {r.get('medication','')} {r.get('note','')}".lower()
        return search in blob
    rows = [r for r in rows if matches(r)]

active = [r for r in rows if r["status"] == "active"]
done = [r for r in rows if r["status"] == "done"]

overdue = []
not_overdue = []
for r in active:
    _, is_overdue = remaining_text(r["due_at"])
    (overdue if is_overdue else not_overdue).append(r)

if view_mode == "Only Active":
    active = not_overdue + overdue
    done = []
    overdue = []
elif view_mode == "Only Overdue":
    active = overdue
    done = []
elif view_mode == "Only Done":
    active = []
    overdue = []
# else All: keep the default tabs below

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

        with st.container(border=True):
            top = st.columns([2.2, 1.15, 1.15, 1.5, 1.5])

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
                        st.caption(f"Done: {fmt_dt_local(r['done_at'])}")
                else:
                    st.error(rem) if is_overdue else st.info(rem)

            with top[4]:
                # safer actions: confirm delete
                if mode == "done":
                    cA, cB = st.columns(2)
                    with cA:
                        if st.button("Undo", key=f"undo_{r['id']}", use_container_width=True):
                            set_done(r["id"], False)
                            st.rerun()
                    with cB:
                        if st.button("Delete", key=f"del_done_{r['id']}", use_container_width=True):
                            delete_row(r["id"])
                            st.rerun()
                else:
                    cA, cB = st.columns(2)
                    with cA:
                        if st.button("Mark done", key=f"done_{r['id']}", use_container_width=True):
                            set_done(r["id"], True)
                            st.rerun()

                        snooze = st.selectbox(
                            "Extend",
                            options=[5, 10, 15, 30, 60, 120],
                            index=0,
                            key=f"snooze_{r['id']}",
                            label_visibility="collapsed",
                        )
                        if st.button(f"+{snooze}m", key=f"ext_{r['id']}", use_container_width=True):
                            extend_due(r["id"], int(snooze))
                            st.rerun()

                    with cB:
                        # "are you sure" delete
                        confirm = st.checkbox("Confirm delete", key=f"confirm_{r['id']}")
                        if st.button("Delete", key=f"del_{r['id']}", disabled=not confirm, use_container_width=True):
                            delete_row(r["id"])
                            st.rerun()


with tab_active:
    st.subheader("Active waiters")
    render_list(active, mode="active")

with tab_overdue:
    st.subheader("Overdue waiters")
    render_list(overdue, mode="overdue")

with tab_done:
    st.subheader("Archive (done)")
    render_list(done, mode="done")
