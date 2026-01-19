# pages/4_Vault.py

import os
import sqlite3
import hashlib
from datetime import datetime, timezone

import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Vault", layout="wide")
st.title("Vault (PIN Locked)")
st.caption("Simple vault: pick a label -> view saved Login ID + Password/Notes. Avoid storing real pharmacy system creds in a public app.")

# -----------------------------
# DB setup (reuse same DB file)
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
            CREATE TABLE IF NOT EXISTS vault_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                login_id TEXT NOT NULL,
                secret_note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

init_db()

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

# -----------------------------
# PIN lock
# -----------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False

with st.sidebar:
    st.header("Vault Lock")

    if "VAULT_PIN" not in st.secrets:
        st.error("Missing VAULT_PIN in Streamlit secrets.")
        st.stop()

    correct_pin_hash = sha256(str(st.secrets["VAULT_PIN"]).strip())

    pin = st.text_input("Enter PIN", type="password", placeholder="PIN")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Unlock", use_container_width=True):
            if sha256(pin.strip()) == correct_pin_hash:
                st.session_state.vault_unlocked = True
                st.success("Unlocked.")
                st.rerun()
            else:
                st.session_state.vault_unlocked = False
                st.error("Wrong PIN.")

    with col2:
        if st.session_state.vault_unlocked:
            if st.button("Lock", use_container_width=True):
                st.session_state.vault_unlocked = False
                st.info("Locked.")
                st.rerun()

    st.divider()
    show_secret = st.toggle("Show password/notes", value=st.session_state.get("show_secret", False))
    st.session_state["show_secret"] = show_secret

if not st.session_state.vault_unlocked:
    st.info("Vault is locked. Enter PIN in the sidebar to unlock.")
    st.stop()

# -----------------------------
# CRUD helpers
# -----------------------------
def fetch_all():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, label, login_id, secret_note, created_at, updated_at
            FROM vault_notes
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]

def add_item(label: str, login_id: str, secret_note: str) -> int:
    now = utc_iso()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO vault_notes (label, login_id, secret_note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (label.strip(), login_id.strip(), secret_note.strip(), now, now),
        )
        conn.commit()
        return int(cur.lastrowid)

def delete_item(item_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM vault_notes WHERE id=?", (item_id,))
        conn.commit()

# -----------------------------
# Add entry (optional, still simple)
# -----------------------------
with st.expander("Add new entry", expanded=False):
    with st.form("add_vault_form", clear_on_submit=True):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            new_label = st.text_input("Label", placeholder="e.g., Canada Life portal")
            new_login = st.text_input("Login ID / Username", placeholder="email / username")
        with c2:
            new_secret = st.text_area("Password / Notes", height=100, placeholder="password / notes...")

        submitted = st.form_submit_button("Save", type="primary")

    if submitted:
        if not new_label.strip() or not new_login.strip() or not new_secret.strip():
            st.error("Need Label + Login ID + Password/Notes.")
        else:
            new_id = add_item(new_label, new_login, new_secret)
            st.success("Saved.")
            st.session_state["selected_vault_id"] = new_id
            st.rerun()

st.divider()

# -----------------------------
# Dropdown -> show selected
# -----------------------------
items = fetch_all()
if not items:
    st.info("No saved entries yet.")
    st.stop()

search = st.text_input("Search label", placeholder="type to filter").strip().lower()
filtered = items if not search else [it for it in items if search in it["label"].lower()]

if not filtered:
    st.info("No entries match that search.")
    st.stop()

# Build dropdown labels
labels = [f"{it['label']}  (#{it['id']})" for it in filtered]
id_by_label = {labels[i]: filtered[i]["id"] for i in range(len(filtered))}

# Default selection
default_index = 0
sel_id = st.session_state.get("selected_vault_id")
if sel_id is not None:
    for i, it in enumerate(filtered):
        if it["id"] == sel_id:
            default_index = i
            break

picked = st.selectbox("Choose saved entry", options=labels, index=default_index)
item_id = id_by_label[picked]
st.session_state["selected_vault_id"] = item_id

item = next(i for i in items if i["id"] == item_id)

st.subheader("Saved info")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Label**")
    st.code(item["label"], language="text")
    st.markdown("**Login ID / Username**")
    st.code(item["login_id"], language="text")

with c2:
    st.markdown("**Password / Notes**")
    st.code(item["secret_note"], language="text")


st.caption(f"Created: {item['created_at']} • Updated: {item['updated_at']}")

# Delete (optional)
st.divider()
colA, colB = st.columns([1, 3])
with colA:
    confirm = st.checkbox("Confirm delete")
with colB:
    if st.button("Delete this entry", disabled=not confirm):
        delete_item(item_id)
        st.success("Deleted.")
        st.session_state["selected_vault_id"] = None
        st.rerun()
