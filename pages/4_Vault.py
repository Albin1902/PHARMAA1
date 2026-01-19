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
st.caption("Stores IDs/password notes locally in SQLite. Use at your own risk. Avoid storing real pharmacy system credentials on a public app.")

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

    # You set this in secrets
    if "VAULT_PIN" not in st.secrets:
        st.error("Missing VAULT_PIN in Streamlit secrets.")
        st.stop()

    correct_pin_hash = sha256(str(st.secrets["VAULT_PIN"]).strip())

    pin = st.text_input("Enter PIN", type="password", placeholder="PIN")
    if st.button("Unlock", use_container_width=True):
        if sha256(pin.strip()) == correct_pin_hash:
            st.session_state.vault_unlocked = True
            st.success("Unlocked.")
        else:
            st.session_state.vault_unlocked = False
            st.error("Wrong PIN.")

    if st.session_state.vault_unlocked:
        if st.button("Lock", use_container_width=True):
            st.session_state.vault_unlocked = False
            st.info("Locked.")

if not st.session_state.vault_unlocked:
    st.info("Vault is locked. Enter PIN in the sidebar to unlock.")
    st.stop()

# -----------------------------
# CRUD functions
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

def add_item(label: str, login_id: str, secret_note: str):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO vault_notes (label, login_id, secret_note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (label.strip(), login_id.strip(), secret_note.strip(), now, now),
        )
        conn.commit()

def update_item(item_id: int, label: str, login_id: str, secret_note: str):
    now = utc_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE vault_notes
            SET label=?, login_id=?, secret_note=?, updated_at=?
            WHERE id=?
            """,
            (label.strip(), login_id.strip(), secret_note.strip(), now, item_id),
        )
        conn.commit()

def delete_item(item_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM vault_notes WHERE id=?", (item_id,))
        conn.commit()

# -----------------------------
# Add new
# -----------------------------
st.subheader("Add new entry")

c1, c2 = st.columns([1, 1])
with c1:
    new_label = st.text_input("Label (where is this used?)", placeholder="e.g., Canada Life portal")
    new_login = st.text_input("Login ID / Username", placeholder="email / username")
with c2:
    new_secret = st.text_area("Password / Notes (stored as text)", height=110, placeholder="Password, security answer, notes...")

if st.button("Save", type="primary"):
    if not new_label.strip() or not new_login.strip() or not new_secret.strip():
        st.error("Need Label + Login ID + Notes.")
    else:
        add_item(new_label, new_login, new_secret)
        st.success("Saved.")
        st.rerun()

st.divider()

# -----------------------------
# List / edit
# -----------------------------
items = fetch_all()
if not items:
    st.info("No vault entries yet.")
    st.stop()

st.subheader("Saved entries")

search = st.text_input("Search", placeholder="filter by label or login id").strip().lower()
filtered = items
if search:
    filtered = [
        it for it in items
        if search in (it["label"] + " " + it["login_id"]).lower()
    ]

options = {f"{it['label']}  •  {it['login_id']}  (#{it['id']})": it["id"] for it in filtered}
pick = st.selectbox("Select an entry", options=list(options.keys()))
item_id = options[pick]
item = next(i for i in items if i["id"] == item_id)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Edit")
    label = st.text_input("Label", value=item["label"], key="e_label")
    login_id = st.text_input("Login ID", value=item["login_id"], key="e_login")
    secret_note = st.text_area("Password / Notes", value=item["secret_note"], height=160, key="e_secret")

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Update", type="primary", use_container_width=True):
            update_item(item_id, label, login_id, secret_note)
            st.success("Updated.")
            st.rerun()

    with b2:
        confirm = st.checkbox("Confirm delete", key="confirm_del")
        if st.button("Delete", use_container_width=True, disabled=not confirm):
            delete_item(item_id)
            st.success("Deleted.")
            st.rerun()

with right:
    st.markdown("### Info")
    st.write(f"**Created:** {item['created_at']}")
    st.write(f"**Updated:** {item['updated_at']}")
    st.caption("This vault is PIN-locked but entries are stored as plain text in SQLite. If you need real security, use encryption or a password manager.")
