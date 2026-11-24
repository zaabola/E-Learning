"""Live terminal view of the channels (rich)."""
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BAR = 24
ROWS = [  # channel, label, how to read it
    ("stress", "Stress", "0.5 = your normal"),
    ("confidence", "Confidence", "0.5 = your normal"),
    ("energy", "Energy", "0.5 = your normal"),
    ("positivity", "Positivity", "0.5 = your normal"),
    ("eye_contact", "Eye contact", "share of last 5 s"),
    ("expressiveness", "Voice variety", "low = monotone"),
    ("pace", "Pace", "0.5 = your normal"),
]


def bar(v: float | None, warn_high: bool = False) -> Text:
    if v is None:
        return Text("  (no data)", style="dim")
    n = int(round(v * BAR))
    colour = "green"
    if warn_high and v > 0.7:
        colour = "red"
    elif not warn_high and v < 0.3:
        colour = "yellow"
    return Text("█" * n, style=colour) + Text("·" * (BAR - n), style="dim") + Text(f" {v:.2f}")


def render(snap: dict, warnings: list[str], coach: dict | None = None) -> Panel:
    ch = snap["channels"]
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="bold")
    t.add_column()
    t.add_column(style="dim")
    for key, label, note in ROWS:
        t.add_row(label, bar(ch.get(key), warn_high=key in ("stress",)), note)

    def fmt(v, unit, d=0):
        return "-" if v is None else f"{v:.{d}f} {unit}"
    facts = Text.assemble(
        ("Pace ", "bold"), fmt(ch.get("pace_wpm"), "wpm"), "   ",
        ("Fillers ", "bold"), fmt(ch.get("fillers_per_min"), "/min", 1), "   ",
        ("Silence ", "bold"), fmt(ch.get("silence_now_s"), "s", 1), "   ",
        ("Face ", "bold"), " ".join(map(str, snap["face_top_emotion"])) if snap["face_top_emotion"] else "-",
    )
    if snap["calibrated"]:
        status = Text("Calibrated to your voice and face", style="green")
    else:
        status = Text(f"Calibrating: keep talking normally ({snap['calibration_progress']:.0%})", style="yellow")
    parts = [status, t, facts]
    if snap.get("transcript"):
        parts.append(Text(f"“{snap['transcript'][-90:]}”", style="italic dim"))
    if coach:
        tip = coach.get("tip")
        if tip:
            style = {1: "cyan", 2: "yellow", 3: "bold red"}.get(tip["level"], "cyan")
            parts.append(Text.assemble((f"{tip['level_name'].upper()}: {tip['text']}", f"bold {style}")))
            parts.append(Text(f"why: {tip['why']}  (source: {tip['source']}; {tip['choice']})", style="dim"))
        if coach.get("checking") is not None:
            parts.append(Text(f"checking if it helped... {coach['checking']:.0f}s", style="yellow"))
        if coach.get("toast"):
            parts.append(Text(coach["toast"]["text"], style="green" if coach["toast"]["kind"] == "good" else "dim"))
    for w in warnings[-3:]:
        parts.append(Text(f"! {w}", style="red"))
    return Panel(Group(*parts), title=f"Oral coach · live signals · {snap['session_s']:.0f}s", border_style="cyan")
