"""Grade chips and matchup badge HTML generators."""
from __future__ import annotations


def grade_color(grade: int) -> str:
    """Return CSS color variable for a 20-80 scouting grade."""
    if grade >= 70:
        return "var(--tdd-gold)"
    if grade >= 55:
        return "var(--tdd-sage)"
    if grade >= 45:
        return "var(--tdd-cream)"
    return "var(--tdd-slate)"


def hitter_grades_html(stats: dict) -> str:
    """Compact grade chips for a hitter: Con/Pow/Spd/Disc/Fld with CI ranges."""
    labels = [
        ("Con", "grade_hit"), ("Pow", "grade_power"), ("Spd", "grade_speed"),
        ("Disc", "grade_discipline"), ("Fld", "grade_fielding"),
    ]
    parts = []
    for abbr, key in labels:
        v = stats.get(key)
        if v is not None:
            c = grade_color(v)
            lo = stats.get(f"{key}_lo")
            hi = stats.get(f"{key}_hi")
            ci_html = ""
            if lo is not None and hi is not None:
                ci_html = (
                    f'<span style="color:var(--tdd-slate); font-size:0.55rem; '
                    f'opacity:0.7; margin-left:1px;">({lo}-{hi})</span>'
                )
            parts.append(f'<span style="color:{c};">{abbr} {v}{ci_html}</span>')
    if not parts:
        return ""
    return (
        f'<span style="font-size:0.65rem; letter-spacing:0.3px; '
        f'display:inline-flex; gap:0.4rem;">'
        + "".join(parts) + '</span>'
    )


def pitcher_grades_html(proj: dict) -> str:
    """Compact grade chips for a pitcher: Stuff/Cmd/Dur with CI ranges."""
    labels = [
        ("Stuff", "grade_stuff"), ("Cmd", "grade_command"), ("Dur", "grade_durability"),
    ]
    parts = []
    for abbr, key in labels:
        v = proj.get(key)
        if v is not None:
            c = grade_color(v)
            lo = proj.get(f"{key}_lo")
            hi = proj.get(f"{key}_hi")
            ci_html = ""
            if lo is not None and hi is not None:
                ci_html = (
                    f'<span style="color:var(--tdd-slate); font-size:0.55rem; '
                    f'opacity:0.7; margin-left:1px;">({lo}-{hi})</span>'
                )
            parts.append(f'<span style="color:{c};">{abbr} {v}{ci_html}</span>')
    if not parts:
        return ""
    return (
        f'<span style="font-size:0.65rem; letter-spacing:0.3px; '
        f'display:inline-flex; gap:0.4rem;">'
        + "".join(parts) + '</span>'
    )


def matchup_lift_badge_html(k_lift: float, bb_lift: float, hr_lift: float) -> str:
    """Compact matchup badge from precomputed lift values."""
    net = k_lift - 0.5 * bb_lift - 0.5 * hr_lift
    if net > 0.03:
        color_var, label = "--tdd-ember", "Pitcher"
    elif net < -0.03:
        color_var, label = "--tdd-sage", "Hitter"
    else:
        return (
            f'<span style="color:var(--tdd-slate); font-size:0.68rem; '
            f'margin-left:0.3rem;">Even</span>'
        )
    return (
        f'<span style="color:var({color_var}); font-size:0.68rem; '
        f'font-weight:600; margin-left:0.3rem;">{label} '
        f'<span style="font-weight:400; font-size:0.62rem;">'
        f'K {k_lift:+.2f}</span></span>'
    )
