# pages/1_Dispense_Calculator.py

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
    administrations_per_day,
)

st.set_page_config(page_title="Dispense Calculator", layout="centered")
st.title("Dispense Calculator (Days ↔ Qty)")


# -----------------------------
# Options
# -----------------------------
FREQ_OPTIONS_COMMON = [
    "qd", "bid", "tid", "qid", "hs",
    "q2h", "q3h", "q4h", "q6h", "q8h", "q12h",
]
ROUTE_OPTIONS = ["po", "sl", "top", "im", "iv", "sc", "subq", "pr", "od", "os", "ou", "ad", "as", "au"]
TIMING_OPTIONS = ["ac", "pc", "hs", "stat"]
FORM_OPTIONS = ["tab", "cap", "mL", "g"]


def format_number(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def mg_per_ml_from_conc(conc_value: float, conc_unit: str) -> float:
    """
    conc_unit:
      - "mg/mL": conc_value is mg per 1 mL
      - "mg/5mL": conc_value is mg per 5 mL
    """
    if conc_value <= 0:
        return 0.0
    if conc_unit == "mg/mL":
        return conc_value
    if conc_unit == "mg/5mL":
        return conc_value / 5.0
    return 0.0


def get_freq_used_for_calc(freq: str, prn: bool, max_freq_if_prn: str | None) -> str | None:
    if prn:
        return max_freq_if_prn
    return freq


# -----------------------------
# Sidebar UI
# -----------------------------
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

    st.subheader("Dose Type (flexible)")
    dose_mode = st.selectbox(
        "How do you want to calculate the dose?",
        options=[
            "Units per dose (tabs/caps)",
            "Volume per dose (mL or g per dose)",
            "Liquid by mg per dose (convert to mL)",
            "Weight-based mg/kg/dose (convert to mL)",
            "Weight-based mg/kg/day (convert to mL)",
        ],
        index=0
    )

    st.subheader("Form / Unit")
    form = st.selectbox("Dispensing unit", options=FORM_OPTIONS, index=1, format_func=fmt_form)

    # Dose inputs depend on mode
    dose_qty = 1.0
    computed_notes = {}  # used later for logic display

    if dose_mode == "Units per dose (tabs/caps)":
        dose_qty = st.number_input("Dose per administration (e.g., 1 cap)", min_value=0.0, value=1.0, step=0.5)

    elif dose_mode == "Volume per dose (mL or g per dose)":
        dose_qty = st.number_input(f"Amount per dose ({form})", min_value=0.0, value=5.0, step=0.5)

    elif dose_mode == "Liquid by mg per dose (convert to mL)":
        # force form to mL if user wants
        mg_dose = st.number_input("mg per dose", min_value=0.0, value=250.0, step=10.0)
        conc_unit = st.selectbox("Concentration unit", options=["mg/5mL", "mg/mL"], index=0)
        conc_value = st.number_input(f"Concentration value ({conc_unit})", min_value=0.0, value=250.0, step=10.0)

        mg_ml = mg_per_ml_from_conc(conc_value, conc_unit)
        ml_per_dose = (mg_dose / mg_ml) if mg_ml > 0 else 0.0

        dose_qty = ml_per_dose
        form = "mL"  # dispensing in mL makes sense here

        computed_notes = {
            "mode": "mg_per_dose",
            "mg_dose": mg_dose,
            "conc_unit": conc_unit,
            "conc_value": conc_value,
            "mg_per_ml": mg_ml,
            "ml_per_dose": ml_per_dose,
        }

    elif dose_mode == "Weight-based mg/kg/dose (convert to mL)":
        weight_kg = st.number_input("Patient weight (kg)", min_value=0.0, value=20.0, step=0.5)
        mg_per_kg_dose = st.number_input("mg/kg per dose", min_value=0.0, value=10.0, step=0.5)

        conc_unit = st.selectbox("Concentration unit", options=["mg/5mL", "mg/mL"], index=0)
        conc_value = st.number_input(f"Concentration value ({conc_unit})", min_value=0.0, value=250.0, step=10.0)

        mg_ml = mg_per_ml_from_conc(conc_value, conc_unit)
        mg_dose = weight_kg * mg_per_kg_dose
        ml_per_dose = (mg_dose / mg_ml) if mg_ml > 0 else 0.0

        dose_qty = ml_per_dose
        form = "mL"

        computed_notes = {
            "mode": "mgkg_per_dose",
            "weight_kg": weight_kg,
            "mg_per_kg_dose": mg_per_kg_dose,
            "mg_dose": mg_dose,
            "conc_unit": conc_unit,
            "conc_value": conc_value,
            "mg_per_ml": mg_ml,
            "ml_per_dose": ml_per_dose,
        }

    elif dose_mode == "Weight-based mg/kg/day (convert to mL)":
        weight_kg = st.number_input("Patient weight (kg)", min_value=0.0, value=20.0, step=0.5)
        mg_per_kg_day = st.number_input("mg/kg per day", min_value=0.0, value=30.0, step=1.0)

        conc_unit = st.selectbox("Concentration unit", options=["mg/5mL", "mg/mL"], index=0)
        conc_value = st.number_input(f"Concentration value ({conc_unit})", min_value=0.0, value=250.0, step=10.0)

        # Note: mg/kg/day needs admins/day from frequency — handled later after freq chosen.
        form = "mL"
        computed_notes = {
            "mode": "mgkg_per_day",
            "weight_kg": weight_kg,
            "mg_per_kg_day": mg_per_kg_day,
            "conc_unit": conc_unit,
            "conc_value": conc_value,
            "mg_per_ml": mg_per_ml_from_conc(conc_value, conc_unit),
        }

    st.subheader("Directions")
    route = st.selectbox("Route", options=ROUTE_OPTIONS, index=0, format_func=fmt_route)

    st.subheader("Frequency (more flexible)")
    freq_mode = st.radio("Pick frequency style", ["Common", "Custom times/day"], horizontal=True)

    if freq_mode == "Common":
        freq = st.selectbox("Frequency", options=FREQ_OPTIONS_COMMON, index=1, format_func=fmt_freq)
    else:
        custom_times = st.number_input("Times per day (e.g., 1 = once daily)", min_value=0.0, value=1.0, step=0.5)
        # We'll store as a special internal marker; we’ll compute admins/day directly.
        freq = "__CUSTOM__"
        computed_notes["custom_times_per_day"] = float(custom_times)

    timing = st.multiselect("Timing (optional)", options=TIMING_OPTIONS, default=[], format_func=fmt_timing)

    st.subheader("PRN")
    prn = st.toggle("PRN (as needed)", value=False)
    indication = None
    max_freq_if_prn = None
    if prn:
        indication = st.text_input("PRN indication (e.g., pain, nausea)", value="").strip() or None
        max_freq_if_prn = st.selectbox(
            "MAX PRN frequency (required for calculations)",
            options=FREQ_OPTIONS_COMMON,
            index=5,
            format_func=fmt_freq,
        )

    st.subheader("Rounding")
    rounding_mode = st.selectbox(
        "Rounding suggestion for quantity",
        options=[
            "No rounding (show exact)",
            "Round up to whole unit (tabs/caps)",
            "Round up to step (mL/g)",
        ],
        index=1,
    )

    step = 1.0
    if rounding_mode == "Round up to step (mL/g)":
        step = st.selectbox("Step size", options=[0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0], index=1)


# -----------------------------
# Resolve admins/day (supports custom)
# -----------------------------
def resolve_admins_per_day(freq_code: str, prn_on: bool, max_prn: str | None, custom_times_per_day: float | None) -> float | None:
    if prn_on:
        if not max_prn:
            return None
        return administrations_per_day(max_prn)

    if freq_code == "__CUSTOM__":
        return float(custom_times_per_day) if custom_times_per_day is not None else None

    return administrations_per_day(freq_code)


freq_used = get_freq_used_for_calc(freq, prn, max_freq_if_prn)
custom_times_per_day = computed_notes.get("custom_times_per_day")
apd = resolve_admins_per_day(freq, prn, max_freq_if_prn, custom_times_per_day)

# If mg/kg/day mode, we now can compute mL per dose using apd
if computed_notes.get("mode") == "mgkg_per_day":
    if apd and apd > 0:
        mg_day = computed_notes["weight_kg"] * computed_notes["mg_per_kg_day"]
        mg_dose = mg_day / apd
        mg_ml = computed_notes["mg_per_ml"]
        ml_per_dose = (mg_dose / mg_ml) if mg_ml > 0 else 0.0
        dose_qty = ml_per_dose

        computed_notes["mg_day"] = mg_day
        computed_notes["mg_dose"] = mg_dose
        computed_notes["ml_per_dose"] = ml_per_dose
    else:
        dose_qty = 0.0  # cannot compute without admins/day


# -----------------------------
# Build Sig (engine-based calc)
# -----------------------------
sig = Sig(
    dose_qty=float(dose_qty),
    form=str(form),
    route=str(route),
    freq=str(freq if freq != "__CUSTOM__" else "qd"),  # sig rendering uses codes; custom is separate display below
    timing=list(timing) if timing else [],
    prn=bool(prn),
    indication=indication,
    max_freq_if_prn=max_freq_if_prn,
)

# -------- Label preview --------
st.subheader("Bottle Label / Sig Preview")
if display_mode == "Full":
    st.write("**Full:** " + render_sig(sig, mode="full"))
    st.write("**Abbrev:** `" + render_sig(sig, mode="abbrev") + "`")
else:
    st.write("**Abbrev:** `" + render_sig(sig, mode="abbrev") + "`")
    st.write("**Full:** " + render_sig(sig, mode="full"))

if freq == "__CUSTOM__":
    st.caption(f"Custom frequency: {format_number(custom_times_per_day)} time(s)/day (used for calculations).")

if computed_notes.get("mode") in {"mg_per_dose", "mgkg_per_dose", "mgkg_per_day"}:
    st.caption(f"Computed dose: {format_number(dose_qty)} mL per dose (used for calculations).")

st.divider()

tab1, tab2 = st.tabs(["Qty Needed (Days → Qty)", "Days Supply (Qty → Days)"])


# -----------------------------
# Tab 1: Days -> Qty
# -----------------------------
with tab1:
    st.subheader("Calculate Quantity Needed")
    days = st.number_input("Number of days", min_value=0.0, value=7.0, step=1.0)

    if st.button("Calculate quantity"):
        # If we can't resolve admins/day (e.g., PRN without max, or mg/kg/day without freq), block.
        if apd is None or apd <= 0:
            st.error("Cannot calculate: frequency/admins per day is missing or invalid. If PRN is ON, choose a MAX PRN frequency.")
        elif dose_qty <= 0:
            st.error("Cannot calculate: dose per administration is 0. Fix your dose inputs.")
        else:
            # Use engine's function (it will compute based on freq codes), but for custom we use our own qty:
            if freq == "__CUSTOM__":
                qty_exact = float(days) * float(dose_qty) * float(apd)
                res = {"ok": True, "admins_per_day": apd, "qty_exact": qty_exact, "warnings": []}
                if prn:
                    res["warnings"].append("Calculated using MAX PRN frequency (worst-case use).")
            else:
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

                st.metric("Administrations/day", f'{float(res["admins_per_day"]):.2f}')
                st.metric("Quantity (exact)", f"{qty_exact:.2f} {sig.form}")

                if rounding_mode != "No rounding (show exact)":
                    st.metric("Quantity (rounded up)", f"{qty_suggested:.2f} {sig.form}")

                for w in res.get("warnings", []):
                    st.warning(w)

                # ----- Logic / math display -----
                freq_used_display = max_freq_if_prn if prn else (f"{format_number(custom_times_per_day)} times/day" if freq == "__CUSTOM__" else freq)
                apd_used = float(res["admins_per_day"])
                dose_used = float(dose_qty)
                days_val = float(days)

                base_qty = days_val * dose_used * apd_used

                with st.expander("Calculation details (show the logic)"):
                    st.write("**Inputs used**")
                    st.write(f"- Days: **{format_number(days_val)}**")
                    st.write(f"- Dose per administration: **{format_number(dose_used)} {sig.form}**")
                    st.write(f"- Frequency used: **{freq_used_display}**")
                    if prn:
                        st.write("- PRN: **Yes** → using MAX PRN frequency (worst-case use)")
                    else:
                        st.write("- PRN: **No**")

                    st.write("**Step 1 — administrations per day**")
                    st.write(f"- Admins/day = **{format_number(apd_used)}**")

                    # Weight/liquid extra logic
                    if computed_notes.get("mode") in {"mg_per_dose", "mgkg_per_dose", "mgkg_per_day"}:
                        st.write("**Dose conversion logic (mg → mL)**")
                        st.write(f"- Concentration: **{format_number(computed_notes['mg_per_ml'])} mg/mL**")
                        if "mg_dose" in computed_notes:
                            st.write(f"- mg per dose: **{format_number(computed_notes['mg_dose'])} mg**")
                        if computed_notes.get("mode") == "mgkg_per_day":
                            st.write(f"- mg per day: **{format_number(computed_notes.get('mg_day', 0.0))} mg/day**")
                        st.code(
                            f"mL per dose = mg per dose ÷ (mg/mL) = {format_number(dose_used)} mL",
                            language="text",
                        )

                    st.write("**Step 2 — quantity**")
                    st.write("- Qty = Days × Dose × Admins/day")
                    st.code(
                        f"{format_number(days_val)} × {format_number(dose_used)} × {format_number(apd_used)} = "
                        f"{format_number(base_qty)} {sig.form}",
                        language="text",
                    )

                    if rounding_mode != "No rounding (show exact)":
                        st.write("**Rounding**")
                        if rounding_mode == "Round up to whole unit (tabs/caps)":
                            st.write("- Rounded up to **whole unit**")
                        elif rounding_mode == "Round up to step (mL/g)":
                            st.write(f"- Rounded up to step size: **{step}**")
                        st.code(f"Rounded Qty = {format_number(qty_suggested)} {sig.form}", language="text")


# -----------------------------
# Tab 2: Qty -> Days
# -----------------------------
with tab2:
    st.subheader("Calculate Days Supply From Dispensed Quantity")
    qty_dispensed = st.number_input(
        "Quantity dispensed (same unit as form)",
        min_value=0.0,
        value=30.0,
        step=1.0,
    )

    if st.button("Calculate days supply"):
        if apd is None or apd <= 0:
            st.error("Cannot calculate: frequency/admins per day is missing or invalid. If PRN is ON, choose a MAX PRN frequency.")
        elif dose_qty <= 0:
            st.error("Cannot calculate: dose per administration is 0. Fix your dose inputs.")
        else:
            # custom frequency uses manual math
            if freq == "__CUSTOM__":
                per_day = float(dose_qty) * float(apd)
                days_exact = float(qty_dispensed) / per_day if per_day > 0 else 0.0
                res = {
                    "ok": True,
                    "admins_per_day": apd,
                    "days_supply_exact": days_exact,
                    "days_supply_floor": int(days_exact // 1),
                    "days_supply_ceil": int(days_exact) if float(days_exact).is_integer() else int(days_exact) + 1,
                    "warnings": [],
                }
                if prn:
                    res["warnings"].append("Calculated using MAX PRN frequency (worst-case use).")
            else:
                res = calc_days_supply(sig, float(qty_dispensed))

            if not res["ok"]:
                st.error(res["error"])
                for w in res.get("warnings", []):
                    st.warning(w)
            else:
                st.metric("Administrations/day", f'{float(res["admins_per_day"]):.2f}')
                st.metric("Days supply (exact)", f'{float(res["days_supply_exact"]):.2f}')
                st.write(f'**Floor:** {res["days_supply_floor"]}  |  **Ceil:** {res["days_supply_ceil"]}')
                for w in res.get("warnings", []):
                    st.warning(w)

                # ----- Logic / math display -----
                freq_used_display = max_freq_if_prn if prn else (f"{format_number(custom_times_per_day)} times/day" if freq == "__CUSTOM__" else freq)
                apd_used = float(res["admins_per_day"])
                dose_used = float(dose_qty)
                qty = float(qty_dispensed)
                per_day = dose_used * apd_used

                with st.expander("Calculation details (show the logic)"):
                    st.write("**Inputs used**")
                    st.write(f"- Quantity dispensed: **{format_number(qty)} {sig.form}**")
                    st.write(f"- Dose per administration: **{format_number(dose_used)} {sig.form}**")
                    st.write(f"- Frequency used: **{freq_used_display}**")
                    if prn:
                        st.write("- PRN: **Yes** → using MAX PRN frequency (worst-case use)")
                    else:
                        st.write("- PRN: **No**")

                    st.write("**Step 1 — administrations per day**")
                    st.write(f"- Admins/day = **{format_number(apd_used)}**")

                    st.write("**Step 2 — use per day**")
                    st.code(
                        f"Per day = Dose × Admins/day = {format_number(dose_used)} × {format_number(apd_used)} = "
                        f"{format_number(per_day)} {sig.form}/day",
                        language="text",
                    )

                    st.write("**Step 3 — days supply**")
                    st.code(
                        f"Days = Qty ÷ Per day = {format_number(qty)} ÷ {format_number(per_day)} = "
                        f"{format_number(float(res['days_supply_exact']))}",
                        language="text",
                    )
