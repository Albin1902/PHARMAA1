import streamlit as st
from sigcalc.engine import (
    Sig,
    render_sig,
    calc_qty_needed,
    calc_days_supply,
    freq_to_plain,
    route_to_plain,
    timing_to_plain,
    form_to_plain,
    round_up_to_step,
)

st.set_page_config(page_title="Dispense Calculator", layout="centered")
st.title("Dispense Calculator (Days ↔ Qty)")

FREQ_OPTIONS = ["bid", "tid", "qid", "hs", "q4h", "q6h", "q8h", "q12h"]
ROUTE_OPTIONS = ["po", "sl", "top", "im", "iv", "sc", "subq", "pr", "od", "os", "ou", "ad", "as", "au"]
TIMING_OPTIONS = ["ac", "pc", "hs", "stat"]
FORM_OPTIONS = ["tab", "cap", "mL", "g"]

with st.sidebar:
    st.header("Data Entry")

    display_mode = st.radio("Label display", ["Abbrev", "Full"], horizontal=True)

    def fmt_freq(code: str) -> str:
        return freq_to_plain(code) if display_mode == "Full" else code

    def fmt_route(code: str) -> str:
        return route_to_plain(code) if display_mode == "Full" else code

    def fmt_timing(code: str) -> str:
        return timing_to_plain(code) if display_mode == "Full" else code

    def fmt_form(code: str) -> str:
        return form_to_plain(code) if display_mode == "Full" else code

    st.subheader("Dose")
    dose_qty = st.number_input("Dose per administration", min_value=0.0, value=1.0, step=0.5)
    form = st.selectbox("Form / Unit", options=FORM_OPTIONS, index=1, format_func=fmt_form)

    st.subheader("Directions")
    route = st.selectbox("Route", options=ROUTE_OPTIONS, index=0, format_func=fmt_route)
    freq = st.selectbox("Frequency", options=FREQ_OPTIONS, index=0, format_func=fmt_freq)
    timing = st.multiselect("Timing (optional)", options=TIMING_OPTIONS, default=[], format_func=fmt_timing)

    # IMPORTANT: STAT extra dose handling
    stat_extra_doses = 0
    if "stat" in [t.lower() for t in timing]:
        stat_extra_doses = st.number_input("STAT extra dose(s) to add", min_value=0, value=1, step=1)

    st.subheader("PRN")
    prn = st.toggle("PRN (as needed)", value=False)
    indication = None
    max_freq_if_prn = None
    if prn:
        indication = st.text_input("PRN indication (e.g., pain, nausea)", value="").strip() or None
        max_freq_if_prn = st.selectbox(
            "MAX PRN frequency (required for calculations)",
            options=FREQ_OPTIONS,
            index=1,
            format_func=fmt_freq
        )

    st.subheader("Rounding")
    rounding_mode = st.selectbox(
        "Rounding suggestion for quantity",
        options=[
            "No rounding (show exact)",
            "Round up to whole unit (tabs/caps)",
            "Round up to step (mL/g)"
        ],
        index=1
    )
    step = 1.0
    if rounding_mode == "Round up to step (mL/g)":
        step = st.selectbox("Step size", options=[0.5, 1.0, 2.0, 5.0, 10.0], index=1)

sig = Sig(
    dose_qty=float(dose_qty),
    form=str(form),
    route=str(route),
    freq=str(freq),
    timing=list(timing) if timing else [],
    prn=bool(prn),
    indication=indication,
    max_freq_if_prn=max_freq_if_prn,
    stat_extra_doses=int(stat_extra_doses),
)

st.subheader("Bottle Label / Sig Preview")
if display_mode == "Full":
    st.write("**Full:** " + render_sig(sig, mode="full"))
    st.write("**Abbrev:** `" + render_sig(sig, mode="abbrev") + "`")
else:
    st.write("**Abbrev:** `" + render_sig(sig, mode="abbrev") + "`")
    st.write("**Full:** " + render_sig(sig, mode="full"))

st.divider()

tab1, tab2 = st.tabs(["Qty Needed (Days → Qty)", "Days Supply (Qty → Days)"])

with tab1:
    st.subheader("Calculate Quantity Needed")
    days = st.number_input("Number of days", min_value=0.0, value=7.0, step=1.0)

    if st.button("Calculate quantity"):
        res = calc_qty_needed(sig, float(days))
        if not res["ok"]:
            st.error(res["error"])
            for w in res.get("warnings", []):
                st.warning(w)
        else:
            qty_exact = float(res["qty_exact"])
            qty_suggested = qty_exact

            if rounding_mode == "Round up to whole unit (tabs/caps)":
                qty_suggested = round_up_to_step(qty_exact, 1.0)
            elif rounding_mode == "Round up to step (mL/g)":
                qty_suggested = round_up_to_step(qty_exact, float(step))

            st.metric("Administrations/day", f'{res["admins_per_day"]:.2f}')
            st.metric("Quantity (exact)", f"{qty_exact:.2f} {sig.form}")
            if rounding_mode != "No rounding (show exact)":
                st.metric("Quantity (rounded up)", f"{qty_suggested:.2f} {sig.form}")

            for w in res.get("warnings", []):
                st.warning(w)

with tab2:
    st.subheader("Calculate Days Supply From Dispensed Quantity")
    qty_dispensed = st.number_input(
        "Quantity dispensed (same unit as form)",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    if st.button("Calculate days supply"):
        res = calc_days_supply(sig, float(qty_dispensed))
        if not res["ok"]:
            st.error(res["error"])
            for w in res.get("warnings", []):
                st.warning(w)
        else:
            st.metric("Administrations/day", f'{res["admins_per_day"]:.2f}')
            st.metric("Days supply (exact)", f'{res["days_supply_exact"]:.2f}')
            st.write(f'**Floor:** {res["days_supply_floor"]}  |  **Ceil:** {res["days_supply_ceil"]}')
            for w in res.get("warnings", []):
                st.warning(w)
