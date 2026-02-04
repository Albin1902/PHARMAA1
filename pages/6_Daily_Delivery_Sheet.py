import os
import sqlite3
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
st.set_page_config(page_title="Daily Delivery Sheet", layout="wide")

PIN_VALUE = str(st.secrets.get("BP_PIN", "2026"))

if "bp_unlocked" not in st.session_state:
    st.session_state.bp_unlocked = False

with st.sidebar:
    st.markdown("### 🔒 Delivery Sheet Lock")
    pin_in = st.text_input("Enter PIN", type="password", placeholder="PIN")
    c_unlock, c_lock = st.columns(2)
    if c_unlock.button("Unlock", use_container_width=True):
        st.session_state.bp_unlocked = (pin_in == PIN_VALUE)
        st.success("Unlocked.") if st.session_state.bp_unlocked else st.error("Wrong PIN.")
    if c_lock.button("Lock", use_container_width=True):
        st.session_state.bp_unlocked = False
        st.info("Locked.")

if not st.session_state.bp_unlocked:
    st.title("Daily Delivery Sheet (BP + RX)")
    st.warning("Locked. Enter PIN to access this page.", icon="🔒")
    st.stop()


# =========================
# DB setup (same DB as tracker)
# =========================
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA busy_timeout=5000;")
    return c


def table_columns(c, table_name: str) -> set[str]:
    rows = c.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def init_db():
    with conn() as c:
        # Base table (must exist)
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                weekday INTEGER NOT NULL,
                interval_weeks INTEGER NOT NULL,
                packs_per_delivery INTEGER NOT NULL,
                anchor_date TEXT NOT NULL,
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        # Auto-migrate extra fields (safe, no wipe)
        cols = table_columns(c, "bp_patients")
        if "address" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN address TEXT")
        if "packages_per_delivery" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN packages_per_delivery INTEGER")
        if "charge_code" not in cols:
            c.execute("ALTER TABLE bp_patients ADD COLUMN charge_code TEXT")

        c.execute("UPDATE bp_patients SET address = COALESCE(address, '')")
        c.execute("UPDATE bp_patients SET packages_per_delivery = COALESCE(packages_per_delivery, 1)")
        c.execute("UPDATE bp_patients SET charge_code = COALESCE(charge_code, '0')")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,
                packs INTEGER,
                note TEXT
            )
            """
        )
        c.commit()


init_db()

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, weekday, interval_weeks, packs_per_delivery,
                   anchor_date, notes, active,
                   address, packages_per_delivery, charge_code
            FROM bp_patients
            ORDER BY active DESC, weekday ASC, interval_weeks ASC, name ASC
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    for col in ["notes", "address", "charge_code"]:
        df[col] = df[col].fillna("").astype(str)
    df["packages_per_delivery"] = pd.to_numeric(df["packages_per_delivery"], errors="coerce").fillna(1).astype(int)
    return df


def read_overrides_for_day(d: date) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, odate, patient_name, action, packs, note
            FROM bp_overrides
            WHERE odate = ?
            ORDER BY patient_name ASC
            """,
            c,
            params=(d.isoformat(),),
        )
    if df.empty:
        return df
    df["odate"] = pd.to_datetime(df["odate"]).dt.date
    df["note"] = df["note"].fillna("").astype(str)
    return df


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
    interval_weeks: int  # 1,2,4, 99 manual add
    override_note: str = ""


def build_day_bp_list(d: date, patients_df: pd.DataFrame) -> list[DeliveryItem]:
    """Auto schedule for ONE day (weekdays only)."""
    if patients_df.empty:
        return []

    if d.weekday() > 4:
        # weekends: nothing auto (you can still add manual lines)
        return []

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return []

    todays = active[active["weekday"] == d.weekday()]
    out: list[DeliveryItem] = []

    for _, r in todays.iterrows():
        anchor = r["anchor_date"]
        if isinstance(anchor, pd.Timestamp):
            anchor = anchor.date()
        interval = int(r["interval_weeks"])
        if occurs_on_day(anchor, interval, d):
            out.append(
                DeliveryItem(
                    name=str(r["name"]),
                    packs=int(r["packs_per_delivery"]),
                    interval_weeks=int(interval),
                )
            )

    # weekly -> biweekly -> monthly -> manual (99) then name
    out.sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return out


def apply_overrides_to_day(items: list[DeliveryItem], overrides_df: pd.DataFrame) -> list[DeliveryItem]:
    if overrides_df.empty:
        return items

    out = items[:]
    for _, r in overrides_df.iterrows():
        name = str(r["patient_name"])
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])
        note = str(r.get("note", "") or "").strip()

        if action == "skip":
            out = [x for x in out if x.name != name]
        elif action == "add":
            out.append(DeliveryItem(name=name, packs=packs or 1, interval_weeks=99, override_note=note))

    out.sort(key=lambda x: (x.interval_weeks, x.name.lower()))
    return out


def truncate_to_width(text: str, max_width: float, font_name: str, font_size: int) -> str:
    text = "" if text is None else str(text)
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


def make_daily_pdf_grid(
    delivery_date: date,
    df_lines: pd.DataFrame,
    page_mode: str = "letter",
    fill_page_with_blanks: bool = True,
    extra_blank_rows: int = 10,
) -> bytes:
    """
    Daily PDF grid for hand-writing:
      - Auto-filled BP rows + manual RX rows
      - Keeps blank row lines to write more
      - Packages + Charged ALWAYS left blank on print (as requested)
      - BP col short, Notes shorter, Address wide
    """
    pagesize = landscape(letter if page_mode == "letter" else legal)
    w, h = pagesize

    tmp_path = os.path.join(DATA_DIR, "_tmp_daily_grid.pdf")
    c = canvas.Canvas(tmp_path, pagesize=pagesize)

    margin = 0.35 * inch
    left, right = margin, w - margin
    bottom, top = margin, h - margin

    title = f"Delivery Sheet — {delivery_date.isoformat()} (BP + RX)"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top - 0.15 * inch, title)
    c.setFont("Helvetica", 9)
    c.drawString(left, top - 0.40 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    table_top = top - 0.65 * inch
    table_left = left
    table_right = right
    table_w = table_right - table_left

    # Column widths (sum = 1.00)
    # Type | BP | Patient | Address | Notes | Packages | Charged
    col_fracs = [0.06, 0.05, 0.20, 0.37, 0.18, 0.07, 0.07]
    col_w = [table_w * f for f in col_fracs]
    headers = ["Type", "BP", "Patient", "Address", "Notes", "Packages", "Charged"]

    font_name = "Helvetica"
    fs_header = 10
    fs_body = 9
    row_h = 0.30 * inch  # strong lines for writing

    # How many rows fit on one page?
    usable_h = (table_top - bottom)
    header_h = 0.30 * inch
    max_rows = int((usable_h - header_h) // row_h)
    if max_rows < 5:
        max_rows = 5

    # Build rows to print (truncate to max_rows, or add blanks)
    df = df_lines.copy() if df_lines is not None else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=headers)

    # ensure columns exist
    for col in headers:
        if col not in df.columns:
            df[col] = ""

    # leave Packages & Charged blank on print (forced)
    df["Packages"] = ""
    df["Charged"] = ""

    # Add extra blank rows for handwriting
    blanks = pd.DataFrame([{h: "" for h in headers} for _ in range(extra_blank_rows)])
    df = pd.concat([df, blanks], ignore_index=True)

    if fill_page_with_blanks and len(df) < max_rows:
        more = max_rows - len(df)
        df = pd.concat([df, pd.DataFrame([{h: "" for h in headers} for _ in range(more)])], ignore_index=True)

    # We may need multiple pages if too many rows
    rows = df.to_dict("records")

    def draw_page(page_rows: list[dict], page_no: int):
        nonlocal c

        # Header band
        c.setStrokeGray(0.70)
        c.setFillGray(0.92)
        c.rect(table_left, table_top - header_h, table_w, header_h, stroke=1, fill=1)

        c.setFillGray(0.0)
        c.setFont("Helvetica-Bold", fs_header)

        x = table_left
        for i, htxt in enumerate(headers):
            c.drawString(x + 4, table_top - header_h + 8, htxt)
            x += col_w[i]

        # Vertical lines
        x = table_left
        for i in range(len(col_w) + 1):
            c.line(x, table_top, x, bottom)
            if i < len(col_w):
                x += col_w[i]

        # Body rows
        c.setFont(font_name, fs_body)
        y = table_top - header_h

        for r in page_rows:
            # row box lines
            c.line(table_left, y, table_right, y)
            c.line(table_left, y - row_h, table_right, y - row_h)

            # cell text (single line truncated)
            x = table_left
            for ci, key in enumerate(headers):
                val = "" if r.get(key) is None else str(r.get(key))
                # force packages/charged blank already, but keep safe
                if key in ["Packages", "Charged"]:
                    val = ""
                max_w = col_w[ci] - 8
                c.drawString(x + 4, y - 0.20 * inch, truncate_to_width(val, max_w, font_name, fs_body))
                x += col_w[ci]

            y -= row_h

        # bottom border
        c.line(table_left, y, table_right, y)

        # tiny footer page number if multi page
        c.setFont("Helvetica", 8)
        c.drawRightString(right, bottom - 0.10 * inch, f"Page {page_no}")

    # paginate
    page_no = 1
    idx = 0
    per_page = max_rows
    while idx < len(rows):
        page_rows = rows[idx : idx + per_page]
        draw_page(page_rows, page_no)
        idx += per_page
        page_no += 1
        if idx < len(rows):
            c.showPage()
            # redraw title on new page
            c.setFont("Helvetica-Bold", 16)
            c.drawString(left, top - 0.15 * inch, title)
            c.setFont("Helvetica", 9)
            c.drawString(left, top - 0.40 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            # (table_top stays same)

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
# UI
# =========================
st.title("Daily Delivery Sheet (BP + RX)")
st.caption("Auto-fill BP deliveries for a date, then add RX/extra deliveries manually. PDF prints with blank lined rows.")

patients_df = read_patients()

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.2])
with c1:
    dpick = st.date_input("Delivery date", value=date.today())
with c2:
    scope = st.radio("Auto-fill scope", ["All", "Weekly", "Biweekly + Monthly"], index=0, horizontal=True)
with c3:
    paper = st.selectbox("Paper", ["letter", "legal"], index=0)
with c4:
    extra_lines = st.number_input("Extra blank lines", min_value=0, max_value=60, value=12, step=1)

fill_page = st.toggle("Fill page with blank lines", value=True)

# Build auto list
items = build_day_bp_list(dpick, patients_df)
ov = read_overrides_for_day(dpick)
items = apply_overrides_to_day(items, ov)

# Filter scope
def allow(it: DeliveryItem, scope: str) -> bool:
    if scope == "Weekly":
        return it.interval_weeks in {1, 99}
    if scope == "Biweekly + Monthly":
        return it.interval_weeks in {2, 4, 99}
    return it.interval_weeks in {1, 2, 4, 99}

items = [it for it in items if allow(it, scope)]

# Lookup address/notes for auto-fill lines (Packages/Charged intentionally NOT filled)
by_name = {}
if not patients_df.empty:
    for _, r in patients_df.iterrows():
        by_name[str(r["name"])] = r

rows = []
for it in items:
    r = by_name.get(it.name, None)
    addr = "" if r is None else str(r.get("address", "") or "")
    notes = "" if r is None else str(r.get("notes", "") or "")
    if it.override_note:
        notes = (notes + " | " + it.override_note).strip(" |")

    rows.append(
        {
            "Type": "BP",
            "BP": str(it.packs),
            "Patient": it.name,
            "Address": addr,
            "Notes": notes,
            "Packages": "",  # leave blank for handwriting
            "Charged": "",   # leave blank for handwriting
        }
    )

# Add a few default manual rows (RX) so you don't start from empty
for _ in range(6):
    rows.append({"Type": "RX", "BP": "", "Patient": "", "Address": "", "Notes": "", "Packages": "", "Charged": ""})

df_lines = pd.DataFrame(rows, columns=["Type", "BP", "Patient", "Address", "Notes", "Packages", "Charged"])

st.subheader(f"Editable daily list — {dpick.isoformat()} ({scope})")
edited = st.data_editor(
    df_lines,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["BP", "RX"], required=False),
        "BP": st.column_config.TextColumn("BP", help="For BP packs, or leave blank for RX."),
        "Patient": st.column_config.TextColumn("Patient"),
        "Address": st.column_config.TextColumn("Address"),
        "Notes": st.column_config.TextColumn("Notes"),
        "Packages": st.column_config.TextColumn("Packages (write on paper)"),
        "Charged": st.column_config.TextColumn("Charged (cc/0) (write on paper)"),
    },
)

st.caption("Packages + Charged will print BLANK so you can fill by pen.")

pdf_bytes = make_daily_pdf_grid(
    delivery_date=dpick,
    df_lines=edited,
    page_mode=paper,
    fill_page_with_blanks=fill_page,
    extra_blank_rows=int(extra_lines),
)

st.download_button(
    "Download Daily PDF (lined + manual space)",
    data=pdf_bytes,
    file_name=f"daily_delivery_{dpick.isoformat()}.pdf",
    mime="application/pdf",
    type="primary",
)
