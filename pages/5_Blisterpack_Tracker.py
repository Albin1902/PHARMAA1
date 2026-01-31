import os
import sqlite3
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

# PDF
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# =========================
# Config
# =========================
st.set_page_config(page_title="Blister Pack Delivery Sheet", layout="wide")

st.title("Blister Pack Delivery Sheet (Auto Month Generator)")
st.caption(
    "This generates a month delivery sheet from your patient list (Weekly / Biweekly / 4-week). "
    "Calendar view is default. Use overrides for holidays/exceptions."
)

# ⚠️ PHI warning
st.warning(
    "If you print patient names, do NOT run this as a public app. Keep it private. "
    "If you must keep it public, use codes instead of names.",
    icon="⚠️"
)

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "blisterpacks.db")
os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# DB helpers
# =========================
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                weekday INTEGER NOT NULL,                -- 0=Mon ... 6=Sun (we use Mon-Fri)
                interval_weeks INTEGER NOT NULL,         -- 1 / 2 / 4
                packs_per_delivery INTEGER NOT NULL,     -- 1 / 2 / 4
                anchor_date TEXT NOT NULL,               -- ISO yyyy-mm-dd (defines cycle)
                notes TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bp_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odate TEXT NOT NULL,                     -- ISO date (yyyy-mm-dd)
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,                    -- 'skip' or 'add'
                packs INTEGER,                           -- only used for 'add'
                note TEXT
            )
            """
        )

        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_bp_patients_weekday ON bp_patients(weekday)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_bp_overrides_date ON bp_overrides(odate)"
        )
        c.commit()


init_db()


def read_patients() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query(
            """
            SELECT id, name, weekday, interval_weeks, packs_per_delivery, anchor_date, notes, active
            FROM bp_patients
            ORDER BY active DESC, weekday ASC, name ASC
            """,
            c,
        )
    if df.empty:
        return df
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.date
    df["active"] = df["active"].astype(bool)
    return df


def upsert_patients(df: pd.DataFrame):
    # Replaces rows by id; inserts rows with id missing/NaN
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
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Calendar header wants Sun..Sat
SUN_FIRST = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start, end


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
    note: str = ""


def build_month_schedule(year: int, month: int, patients_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    start, end = month_bounds(year, month)
    schedule: dict[date, list[DeliveryItem]] = {d: [] for d in dates_in_month(year, month)}

    if patients_df.empty:
        return schedule

    active = patients_df[patients_df["active"] == True].copy()
    if active.empty:
        return schedule

    for d in schedule.keys():
        # we allow Sun..Sat in calendar, but only generate deliveries for Mon-Fri
        if d.weekday() > 4:  # 5=Sat,6=Sun
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
                        note="" if pd.isna(r.get("notes", "")) else str(r.get("notes", "")),
                    )
                )

        # sort for neatness
        schedule[d].sort(key=lambda x: x.name.lower())

    return schedule


def apply_overrides(schedule: dict[date, list[DeliveryItem]], overrides_df: pd.DataFrame) -> dict[date, list[DeliveryItem]]:
    if overrides_df.empty:
        return schedule

    # Build helper for quick removal
    for _, r in overrides_df.iterrows():
        d = r["odate"]
        name = str(r["patient_name"])
        action = str(r["action"]).lower().strip()
        packs = None if pd.isna(r.get("packs", None)) else int(r["packs"])
        note = "" if pd.isna(r.get("note", "")) else str(r.get("note", ""))

        if d not in schedule:
            continue

        if action == "skip":
            schedule[d] = [x for x in schedule[d] if x.name != name]
        elif action == "add":
            # add even on weekends if you want—BUT your rule says deliveries weekdays only.
            # We'll still allow adding, but you can just avoid using weekends.
            schedule[d].append(DeliveryItem(name=name, packs=packs or 1, note=note))
            schedule[d].sort(key=lambda x: x.name.lower())

    return schedule


# =========================
# PDF Generators (Landscape)
# =========================
def make_month_pdf(year: int, month: int, schedule: dict[date, list[DeliveryItem]], max_names_per_cell: int = 10) -> bytes:
    # landscape letter
    pagesize = landscape(letter)
    w, h = pagesize
    buf_path = os.path.join(DATA_DIR, "_tmp_month.pdf")

    c = canvas.Canvas(buf_path, pagesize=pagesize)
    title = f"Blister Pack Delivery Sheet — {calendar.month_name[month]} {year}"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.6 * inch, h - 0.6 * inch, title)

    c.setFont("Helvetica", 9)
    c.drawString(0.6 * inch, h - 0.85 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # grid area
    top = h - 1.2 * inch
    left = 0.5 * inch
    right = w - 0.5 * inch
    bottom = 0.6 * inch

    grid_w = right - left
    grid_h = top - bottom

    col_w = grid_w / 7.0
    # up to 6 weeks displayed in month grid
    row_h = grid_h / 6.5  # little extra for header row
    header_h = row_h * 0.55
    day_h = (grid_h - header_h) / 6.0

    # header row Sun..Sat
    c.setFillGray(0.92)
    c.rect(left, top - header_h, grid_w, header_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 10)
    for i, lbl in enumerate(SUN_FIRST):
        c.drawString(left + i * col_w + 4, top - header_h + 6, lbl)

    # build weeks with Sunday-first calendar
    cal = calendar.Calendar(firstweekday=6)  # Sunday
    weeks = cal.monthdatescalendar(year, month)  # list of weeks, each is 7 dates
    # pad to 6 weeks
    while len(weeks) < 6:
        last = weeks[-1]
        weeks.append([d + timedelta(days=7) for d in last])

    # draw cells
    c.setFont("Helvetica", 8)
    for r, week in enumerate(weeks[:6]):
        y_top = top - header_h - r * day_h
        y_bot = y_top - day_h

        # horizontal line
        c.setStrokeGray(0.7)
        c.line(left, y_top, right, y_top)

        for col, d in enumerate(week):
            x0 = left + col * col_w
            x1 = x0 + col_w

            # vertical lines
            c.setStrokeGray(0.7)
            c.line(x0, y_bot, x0, y_top)

            # cell background for other month days
            if d.month != month:
                c.setFillGray(0.97)
                c.rect(x0, y_bot, col_w, day_h, stroke=0, fill=1)
                c.setFillGray(0.0)

            # date top-left
            c.setFont("Helvetica-Bold", 9)
            c.drawString(x0 + 4, y_top - 12, str(d.day))

            # list items
            items = schedule.get(d, [])
            # show only up to max
            shown = items[:max_names_per_cell]
            extra = max(0, len(items) - len(shown))

            # dynamic font smaller if crowded
            font_size = 8
            if len(items) >= 8:
                font_size = 7
            if len(items) >= 12:
                font_size = 6
            c.setFont("Helvetica", font_size)

            yy = y_top - 24
            line_h = font_size + 2
            for it in shown:
                # wrap long names crudely (split)
                line = f"{it.name}  ({it.packs}p)"
                if len(line) > 28:
                    line = line[:27] + "…"
                c.drawString(x0 + 4, yy, line)
                yy -= line_h
                if yy < y_bot + 6:
                    break

            if extra > 0 and yy > y_bot + 6:
                c.setFont("Helvetica-Oblique", font_size)
                c.drawString(x0 + 4, yy, f"+{extra} more")

    # final border lines
    c.setStrokeGray(0.7)
    c.line(right, bottom, right, top - header_h)
    c.line(left, bottom, right, bottom)

    c.showPage()
    c.save()

    with open(buf_path, "rb") as f:
        data = f.read()
    try:
        os.remove(buf_path)
    except Exception:
        pass
    return data


def make_week_pdf(week_start: date, schedule: dict[date, list[DeliveryItem]]) -> bytes:
    pagesize = landscape(letter)
    w, h = pagesize
    buf_path = os.path.join(DATA_DIR, "_tmp_week.pdf")

    c = canvas.Canvas(buf_path, pagesize=pagesize)
    week_end = week_start + timedelta(days=6)
    title = f"Blister Pack — Weekly Delivery Sheet ({week_start.isoformat()} to {week_end.isoformat()})"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.6 * inch, h - 0.6 * inch, title)

    c.setFont("Helvetica", 9)
    c.drawString(0.6 * inch, h - 0.85 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    left = 0.6 * inch
    right = w - 0.6 * inch
    top = h - 1.2 * inch
    bottom = 0.6 * inch

    # Table columns
    col_day = 2.2 * inch
    col_due = (right - left) - col_day
    row_h = (top - bottom) / 7.5

    # Header row
    c.setFillGray(0.92)
    c.rect(left, top - row_h, right - left, row_h, stroke=0, fill=1)
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left + 6, top - row_h + 8, "Day")
    c.drawString(left + col_day + 6, top - row_h + 8, "Due (Patient — Packs)   [ ] Delivered   Notes")

    # Rows
    c.setFont("Helvetica", 9)
    y = top - row_h
    for i in range(7):
        d = week_start + timedelta(days=i)
        y2 = y - row_h

        c.setStrokeGray(0.7)
        c.line(left, y, right, y)
        c.line(left, y2, right, y2)
        c.line(left, y2, left, y)
        c.line(left + col_day, y2, left + col_day, y)
        c.line(right, y2, right, y)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(left + 6, y2 + row_h - 14, f"{WEEKDAY_LABELS[d.weekday()]} {d.isoformat()}")

        c.setFont("Helvetica", 9)
        items = schedule.get(d, [])
        yy = y2 + row_h - 14
        line_h = 11
        for it in items[:18]:
            line = f"{it.name} — {it.packs}p"
            if len(line) > 80:
                line = line[:79] + "…"
            c.drawString(left + col_day + 10, yy, line)
            yy -= line_h
            if yy < y2 + 6:
                break

        y = y2

    c.showPage()
    c.save()

    with open(buf_path, "rb") as f:
        data = f.read()
    try:
        os.remove(buf_path)
    except Exception:
        pass
    return data


# =========================
# UI — Tabs
# =========================
tab_cal, tab_patients, tab_overrides, tab_print = st.tabs(
    ["📅 Calendar (default)", "👥 Patients", "✏️ Overrides", "🖨️ Print PDFs"]
)

# Default month selection (today)
today = date.today()
with tab_cal:
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        year = st.number_input("Year", min_value=2020, max_value=2100, value=today.year, step=1)
    with c2:
        month = st.selectbox("Month", list(range(1, 13)), index=today.month - 1, format_func=lambda m: calendar.month_name[m])
    with c3:
        max_per_cell = st.slider("Max names shown per day cell", 3, 20, 10, 1)

    patients_df = read_patients()
    base = build_month_schedule(int(year), int(month), patients_df)
    start, end = month_bounds(int(year), int(month))
    overrides_df = read_overrides(start, end)
    schedule = apply_overrides(base, overrides_df)

    st.subheader(f"{calendar.month_name[int(month)]} {int(year)} (Sun → Sat)")

    # CSS for tighter cell look
    st.markdown(
        """
        <style>
        .bp-cell {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 8px;
            min-height: 140px;
        }
        .bp-date {
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 6px;
        }
        .bp-muted { opacity: 0.45; }
        .bp-item { font-size: 12px; line-height: 1.25; margin: 0 0 2px 0; }
        .bp-more { font-size: 12px; opacity: 0.7; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # month grid (Sunday-first)
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(int(year), int(month))
    while len(weeks) < 6:
        last = weeks[-1]
        weeks.append([d + timedelta(days=7) for d in last])

    # Header row
    hdr = st.columns(7)
    for i, lbl in enumerate(SUN_FIRST):
        hdr[i].markdown(f"**{lbl}**")

    # Render weeks
    for week in weeks[:6]:
        cols = st.columns(7)
        for i, d in enumerate(week):
            items = schedule.get(d, [])
            shown = items[:max_per_cell]
            extra = max(0, len(items) - len(shown))

            muted = "bp-muted" if d.month != int(month) else ""
            weekend = (d.weekday() >= 5)
            # keep weekend visible but empty-ish
            if weekend and d.month == int(month):
                # show date but no deliveries (by design)
                shown = []
                extra = 0

            # build HTML
            lines = []
            for it in shown:
                nm = it.name
                # shorten for UI if super long
                if len(nm) > 26:
                    nm = nm[:25] + "…"
                lines.append(f"<div class='bp-item'>{nm} <span style='opacity:0.7'>({it.packs}p)</span></div>")
            more = f"<div class='bp-more'>+{extra} more</div>" if extra > 0 else ""

            html = f"""
            <div class="bp-cell {muted}">
              <div class="bp-date">{d.day}</div>
              {''.join(lines)}
              {more}
            </div>
            """
            cols[i].markdown(html, unsafe_allow_html=True)

    # List view (optional) under calendar
    st.divider()
    st.subheader("Month list view (grouped by date)")
    for d in dates_in_month(int(year), int(month)):
        if d.weekday() >= 5:
            continue
        items = schedule.get(d, [])
        if not items:
            continue
        with st.expander(f"{WEEKDAY_LABELS[d.weekday()]} — {d.isoformat()} ({len(items)} deliveries)"):
            df = pd.DataFrame([{"Name": it.name, "Packs": it.packs, "Note": it.note} for it in items])
            st.dataframe(df, use_container_width=True)


with tab_patients:
    st.subheader("Patients master list (Add / Edit / Delete)")
    st.caption(
        "Weekly = 1 pack every 1 week, Biweekly = 2 packs every 2 weeks, 4-week = 4 packs every 4 weeks. "
        "Anchor date defines the cycle start."
    )

    freq_map = {
        "Weekly (1 pack)": (1, 1),
        "Biweekly (2 packs)": (2, 2),
        "4-week (4 packs)": (4, 4),
    }

    df = read_patients()
    if df.empty:
        df = pd.DataFrame(
            columns=["id", "name", "weekday", "interval_weeks", "packs_per_delivery", "anchor_date", "notes", "active"]
        )

    # show weekday as label via numeric; user edits via select in a helper column
    df_view = df.copy()
    if not df_view.empty:
        df_view["weekday"] = df_view["weekday"].astype(int)

    st.write("Edit rows below, then click **Save changes**.")
    edited = st.data_editor(
        df_view,
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
            "interval_weeks": st.column_config.SelectboxColumn("Interval (weeks)", options=[1, 2, 4], required=True),
            "packs_per_delivery": st.column_config.SelectboxColumn("Packs per delivery", options=[1, 2, 4], required=True),
            "anchor_date": st.column_config.DateColumn("Anchor date", required=True),
            "notes": st.column_config.TextColumn("Notes"),
            "active": st.column_config.CheckboxColumn("Active"),
        },
    )

    # Guardrail: enforce (interval == packs) by default if user messes it up
    auto_fix = st.toggle("Auto-fix packs to match frequency (recommended)", value=True)
    if auto_fix and not edited.empty:
        for idx in edited.index:
            try:
                interval = int(edited.loc[idx, "interval_weeks"])
                edited.loc[idx, "packs_per_delivery"] = interval
            except Exception:
                pass

    if st.button("Save changes", type="primary"):
        # basic validation
        if not edited.empty:
            bad = edited[edited["name"].astype(str).str.strip() == ""]
            if not bad.empty:
                st.error("Some rows have empty names. Fix them or delete those rows.")
                st.stop()
        upsert_patients(edited)
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.subheader("Delete a patient (safe)")
    del_df = read_patients()
    if del_df.empty:
        st.info("No patients to delete.")
    else:
        options = [f"{r['name']} (ID {r['id']})" for _, r in del_df.iterrows()]
        pick = st.selectbox("Select patient", options)
        pid = int(pick.split("ID ")[1].replace(")", ""))
        confirm = st.checkbox("Confirm delete", value=False)
        if st.button("Delete selected", disabled=not confirm):
            delete_patient_by_id(pid)
            st.success("Deleted.")
            st.rerun()


with tab_overrides:
    st.subheader("Overrides (manual exceptions)")
    st.caption(
        "Use overrides for holidays or patient-specific changes. "
        "‘Skip’ removes that patient from that date. ‘Add’ inserts an extra delivery on that date."
    )

    # Month selector for overrides
    o1, o2 = st.columns([1, 1])
    with o1:
        oy = st.number_input("Override year", 2020, 2100, today.year, 1, key="oy")
    with o2:
        om = st.selectbox("Override month", list(range(1, 13)), index=today.month - 1,
                          format_func=lambda m: calendar.month_name[m], key="om")

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
            st.write(" ")
            st.write(" ")
    with oc5:
        note = st.text_input("Note", placeholder="e.g., called pt ok Monday / pickup")

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


with tab_print:
    st.subheader("Print PDFs (Landscape)")
    st.caption("These PDFs use your master list + overrides.")

    py = st.number_input("Print year", 2020, 2100, today.year, 1, key="py")
    pm = st.selectbox("Print month", list(range(1, 13)), index=today.month - 1,
                      format_func=lambda m: calendar.month_name[m], key="pm")
    pmax = st.slider("Max names per day cell (PDF)", 3, 20, 10, 1)

    patients_df = read_patients()
    base = build_month_schedule(int(py), int(pm), patients_df)
    start, end = month_bounds(int(py), int(pm))
    overrides_df = read_overrides(start, end)
    sched = apply_overrides(base, overrides_df)

    # Month PDF
    pdf_month = make_month_pdf(int(py), int(pm), sched, max_names_per_cell=int(pmax))
    st.download_button(
        "Download Month PDF (Landscape)",
        data=pdf_month,
        file_name=f"bp_month_{py}_{pm:02d}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.divider()

    # Week PDF
    st.markdown("### Weekly PDF")
    any_day = st.date_input("Pick any day in the week to print", value=today, key="weekpick")
    # week start Monday for the weekly sheet (more pharmacy-like)
    week_start = any_day - timedelta(days=any_day.weekday())
    # build schedule for that week: reuse month schedule but only dates needed (or build across boundary)
    # easiest: build a small schedule dict for 7 dates
    week_sched: dict[date, list[DeliveryItem]] = {}
    for i in range(7):
        d = week_start + timedelta(days=i)
        # ensure base schedule available even if week crosses month boundary:
        base_week = build_month_schedule(d.year, d.month, patients_df)
        ov_week = read_overrides(*month_bounds(d.year, d.month))
        base_week = apply_overrides(base_week, ov_week)
        week_sched[d] = base_week.get(d, [])

    pdf_week = make_week_pdf(week_start, week_sched)
    st.download_button(
        f"Download Week PDF (Landscape) — starting {week_start.isoformat()}",
        data=pdf_week,
        file_name=f"bp_week_{week_start.isoformat()}.pdf",
        mime="application/pdf",
    )
