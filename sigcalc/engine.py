from __future__ import annotations

import re
from dataclasses import dataclass
from math import floor, ceil
from typing import Optional, Dict, Any, List

# -----------------------------
# Dictionaries (abbrev -> full)
# -----------------------------
FORM_DISPLAY = {
    "tab": "tablet(s)",
    "cap": "capsule(s)",
    "ml": "mL",
    "g": "gram(s)",
    "gm": "gram(s)",
}

ROUTE_DISPLAY = {
    "po": "by mouth",
    "im": "intramuscularly",
    "iv": "intravenously",
    "sc": "subcutaneously",
    "subq": "subcutaneously",
    "sl": "under the tongue",
    "pr": "rectally",
    "top": "topically",
    "ad": "right ear",
    "as": "left ear",
    "au": "both ears",
    "od": "right eye",
    "os": "left eye",
    "ou": "both eyes",
}

TIMING_DISPLAY = {
    "ac": "before meals",
    "pc": "after meals",
    "hs": "at bedtime",
    "stat": "immediately",
    "prn": "as needed",
    "adlib": "as desired",
    "ad lib": "as desired",
}

FREQ_MAP = {
    "qd": 1,
    "daily": 1,
    "hs": 1,
    "bid": 2,
    "tid": 3,
    "qid": 4,
}

FREQ_DISPLAY = {
    "qd": "once daily",
    "daily": "once daily",
    "hs": "at bedtime",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
}

_QH_RE = re.compile(r"^q(\d+)\s*(h|hr)$", re.IGNORECASE)


def normalize_code(s: str) -> str:
    return (s or "").strip().lower()


# -----------------------------
# Frequency parsing
# -----------------------------
def administrations_per_day(freq: str) -> Optional[float]:
    """
    Convert frequency code (e.g., 'bid', 'hs', 'q6h') into administrations/day.
    Returns None if it cannot be determined.
    """
    f = normalize_code(freq).replace(" ", "")
    if not f:
        return None

    if f in FREQ_MAP:
        return float(FREQ_MAP[f])

    m = _QH_RE.match(f)
    if m:
        hours = int(m.group(1))
        if hours <= 0:
            return None
        return 24.0 / hours

    return None


def freq_to_plain(freq: str) -> str:
    f = normalize_code(freq).replace(" ", "")
    if not f:
        return ""

    if f in FREQ_DISPLAY:
        return FREQ_DISPLAY[f]

    m = _QH_RE.match(f)
    if m:
        hours = int(m.group(1))
        return f"every {hours} hours"

    return f


def route_to_plain(route: str) -> str:
    r = normalize_code(route)
    return ROUTE_DISPLAY.get(r, r)


def timing_to_plain(code: str) -> str:
    t = normalize_code(code)
    return TIMING_DISPLAY.get(t, t)


def form_to_plain(form: str) -> str:
    f = normalize_code(form)
    return FORM_DISPLAY.get(f, f)


# -----------------------------
# Rounding helper
# -----------------------------
def round_up_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return ceil(value / step) * step


# -----------------------------
# Sig model (for label directions)
# -----------------------------
@dataclass(frozen=True)
class Sig:
    dose_qty: float                 # e.g., 1 (per administration)
    form: str                       # tab/cap/ml/g
    route: str                      # PO/SL/Top/etc (store code)
    freq: str                       # bid/q6h/etc (store code)
    timing: Optional[List[str]] = None   # ["pc"] etc
    prn: bool = False
    indication: Optional[str] = None
    max_freq_if_prn: Optional[str] = None  # for PRN calculations (e.g., q6h)


def render_sig(sig: Sig, mode: str = "abbrev") -> str:
    """
    mode:
      - "abbrev": returns abbreviations (fast entry)
      - "full": returns full plain-English style
    """
    mode = normalize_code(mode)
    timing_list = sig.timing or []

    if mode == "full":
        dose_part = f"Take {sig.dose_qty:g} {form_to_plain(sig.form)}"
        route_part = route_to_plain(sig.route) if sig.route else ""
        freq_part = freq_to_plain(sig.freq) if sig.freq else ""

        timing_part = ""
        if timing_list:
            timing_part = " " + ", ".join(timing_to_plain(x) for x in timing_list)

        prn_part = ""
        if sig.prn:
            prn_part = " as needed"
            if sig.indication:
                prn_part += f" for {sig.indication}"

        # Example: "Take 1 tablet(s) by mouth twice daily after meals as needed for pain"
        parts = [dose_part]
        if route_part:
            parts.append(route_part)
        if freq_part:
            parts.append(freq_part)

        out = " ".join(parts) + timing_part + prn_part
        return out.strip()

    # abbrev
    dose_part = f"{sig.dose_qty:g} {normalize_code(sig.form)}".strip()
    route_part = normalize_code(sig.route)
    freq_part = normalize_code(sig.freq)

    timing_part = " ".join(normalize_code(x) for x in timing_list).strip()

    prn_part = ""
    if sig.prn:
        prn_part = "prn"
        if sig.indication:
            prn_part += f" {sig.indication}"

    chunks = [c for c in [dose_part, route_part, freq_part, timing_part, prn_part] if c]
    return " ".join(chunks).strip()


# -----------------------------
# Calculator (days <-> quantity)
# -----------------------------
def calc_qty_needed(sig: Sig, days: float) -> Dict[str, Any]:
    """
    Quantity needed for given days:
      qty = days * dose_qty * administrations_per_day(freq)
    If PRN: requires max_freq_if_prn to compute worst-case qty.
    """
    if days <= 0:
        return {"ok": False, "error": "days must be > 0", "warnings": []}
    if sig.dose_qty <= 0:
        return {"ok": False, "error": "dose_qty must be > 0", "warnings": []}

    warnings = []
    freq_for_calc = sig.freq

    if sig.prn:
        if not sig.max_freq_if_prn:
            return {
                "ok": False,
                "error": "PRN needs a max frequency (e.g., q6h) to calculate quantity safely.",
                "warnings": ["PRN without max frequency: cannot compute quantity."],
            }
        freq_for_calc = sig.max_freq_if_prn
        warnings.append("Calculated using MAX PRN frequency (worst-case use).")

    apd = administrations_per_day(freq_for_calc)
    if apd is None:
        return {"ok": False, "error": f"Unrecognized frequency: '{freq_for_calc}'", "warnings": []}

    qty_exact = days * sig.dose_qty * apd

    return {
        "ok": True,
        "admins_per_day": apd,
        "qty_exact": qty_exact,
        "warnings": warnings,
    }


def calc_days_supply(sig: Sig, qty_dispensed: float) -> Dict[str, Any]:
    """
    Days supply from quantity dispensed:
      days = qty_dispensed / (dose_qty * administrations_per_day(freq))
    If PRN: requires max_freq_if_prn to compute max-use days supply.
    """
    if qty_dispensed <= 0:
        return {"ok": False, "error": "qty_dispensed must be > 0", "warnings": []}
    if sig.dose_qty <= 0:
        return {"ok": False, "error": "dose_qty must be > 0", "warnings": []}

    warnings = []
    freq_for_calc = sig.freq

    if sig.prn:
        if not sig.max_freq_if_prn:
            return {
                "ok": False,
                "error": "PRN needs a max frequency (e.g., q6h) to calculate days supply safely.",
                "warnings": ["PRN without max frequency: cannot compute days supply."],
            }
        freq_for_calc = sig.max_freq_if_prn
        warnings.append("Calculated using MAX PRN frequency (worst-case use).")

    apd = administrations_per_day(freq_for_calc)
    if apd is None:
        return {"ok": False, "error": f"Unrecognized frequency: '{freq_for_calc}'", "warnings": []}

    per_day = sig.dose_qty * apd
    days_exact = qty_dispensed / per_day

    return {
        "ok": True,
        "admins_per_day": apd,
        "per_day": per_day,
        "days_supply_exact": days_exact,
        "days_supply_floor": floor(days_exact),
        "days_supply_ceil": ceil(days_exact),
        "warnings": warnings,
    }
