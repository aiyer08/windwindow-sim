"""
Shared figure style.

One place for colours and typography so every figure in the report reads as
one system. The categorical palette below was run through the data-viz
validator (light surface #fcfcfb): all six checks pass on the first four
slots. Slots 3 and 4 fall below 3:1 contrast on the light surface, so every
chart that uses them also carries a legend plus a direct label, and the
numbers behind every figure are written out as a CSV table in results/tables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write PNG files; never open a window

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

# --- chart chrome & ink (light mode) --------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots, in fixed order (never cycled) ---------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Fixed role -> slot assignments. Colour follows the entity, so these never
# shift when a chart happens to show fewer series.
POLICY_COLOR = {
    "timer": SERIES[1],        # orange
    "simple_rule": SERIES[3],  # yellow
    "model": SERIES[0],        # blue
    "random": INK_MUTED,
}
POLICY_LABEL = {
    "timer": "Timer",
    "simple_rule": "Baseline rule",
    "model": "Model",
    "random": "Randomised (training)",
}
ACTION_COLOR = {
    "shut": SERIES[0],      # blue
    "open": SERIES[1],      # orange
    "open_fan": SERIES[2],  # aqua
}
ACTION_LABEL = {
    "shut": "Window shut",
    "open": "Window open",
    "open_fan": "Open + fan",
}
TEMP_COLOR = {
    "t_in": SERIES[0],    # blue
    "t_wall": SERIES[1],  # orange
    "t_out": SERIES[2],   # aqua
}
TEMP_LABEL = {"t_in": "Indoor air", "t_wall": "Shell", "t_out": "Outdoor air"}

SCENARIO_LABEL = {
    "cool": "Cool outside",
    "hot": "Hot outside",
    "hot_then_cooling": "Hot, then cooling",
    "hot_walls_cold_outside": "Warm shell, cooler outside",
}

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "text.color": INK,
    # Recessive axes: no box, hairline baseline only.
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 8,
    "axes.titlecolor": INK,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    # Recessive grid: hairline, behind the data.
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.labelcolor": INK_MUTED,
    "ytick.labelcolor": INK_MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    # 2 px data lines at 150 dpi (1 pt = 2.08 px).
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.labelcolor": INK_2,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4,
}


def use_style() -> None:
    """Apply the shared style. Call once at the top of any plotting script."""
    plt.rcParams.update(_RC)


def direct_label(ax, x, y, text, color, *, dx=6, dy=0, va="center", ha="left"):
    """Label a series at its end: an 8 px coloured dot, then ink text.

    The dot carries the identity; the text stays in ink, so identity is never
    carried by the colour of the words themselves.
    """
    ax.plot([x], [y], marker="o", ms=4.0, color=color, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=0.7, clip_on=False)
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(dx, dy),
        color=INK_2, fontsize=7.5, va=va, ha=ha, annotation_clip=False,
    )


def legend_below(ax, *, ncol=4, gap=30.0, source=None):
    """Put the legend in one row `gap` points below the bottom of the axes.

    Below, not inside and not above: with direct labels at the right-hand end
    of every series an inside legend collides with them, and an above-axes
    legend fights matplotlib's automatic title placement.

    The anchor is an offset in points from the axes corner, so it survives a
    later tight_layout() call -- an offset expressed as an axes fraction does
    not, because tight_layout resizes the axes underneath it.
    """
    trans = offset_copy(ax.transAxes, fig=ax.figure, x=0.0, y=-gap, units="points")
    # `source` lets a legend sit under one panel while naming the series drawn
    # on another (or on several) -- used where a run has a tall temperature
    # panel above a short action panel and the legend belongs at the bottom.
    args = ()
    if source is not None:
        panels = source if isinstance(source, (list, tuple)) else [source]
        handles, labels = [], []
        for panel in panels:
            h, l = panel.get_legend_handles_labels()
            handles += h
            labels += l
        args = (handles, labels)
    return ax.legend(
        *args, loc="upper left", bbox_to_anchor=(0.0, 0.0), bbox_transform=trans,
        ncol=ncol, borderaxespad=0.0, frameon=False,
    )


def caption(ax, text, *, gap=52.0):
    """A one-line note `gap` points below the bottom of the axes, in muted ink."""
    return ax.annotate(
        text, xy=(0.0, 0.0), xycoords="axes fraction",
        textcoords="offset points", xytext=(0.0, -gap),
        color=INK_MUTED, fontsize=6.8, va="top", ha="left",
        annotation_clip=False, linespacing=1.5,
    )


def save(fig, path, *, note=None, ax=None, note_gap=52.0):
    if note is not None:
        target = ax if ax is not None else fig.axes[-1]
        extra = [caption(target, note, gap=note_gap)]
    else:
        extra = None
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18, bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_session(df, title, *, note=None, path=None, figsize=(6.8, 4.2),
                 comfort=26.0, floor=None, measured=True):
    """The standard two-panel picture of one simulation run.

    Top: the three temperatures. Bottom: what the window and fan were doing.

    The action traces are drawn in ink, not in a series colour. The series
    colours already stand for the three temperatures in the panel above, and
    reusing one of them for "window position" would make the same colour mean
    two different things in one figure.
    """
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    t = df["t_s"] / 60.0
    suffix = " (measured)" if measured else ""
    ax = axes[0]
    ax.axhline(comfort, color=AXIS, lw=0.5, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"{comfort:.0f} °C", xy=(t.iloc[-1] * 1.01, comfort),
                color=INK_MUTED, fontsize=6.4, va="center")
    if floor is not None:
        ax.axhline(floor, color=AXIS, lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.annotate(f"{floor:.0f} °C floor", xy=(t.iloc[-1] * 1.01, floor),
                    color=INK_MUTED, fontsize=6.4, va="center")
    cols = ("t_in", "t_wall", "t_out") if measured else ("t_in_true", "t_wall_true", "t_out_true")
    for col, key in zip(cols, ("t_in", "t_wall", "t_out")):
        ax.plot(t, df[col], color=TEMP_COLOR[key], label=TEMP_LABEL[key], zorder=4)
        direct_label(ax, t.iloc[-1], df[col].iloc[-1], TEMP_LABEL[key], TEMP_COLOR[key], dx=5)
    ax.set_ylabel(f"°C{suffix}")
    ax.set_title(title)
    ax.set_xlim(0, t.iloc[-1] * 1.18)

    ax2 = axes[1]
    ax2.fill_between(t, 0, df["u"], step="post", color=GRID, zorder=2)
    ax2.plot(t, df["u"], drawstyle="steps-post", color=INK_2, lw=1.0, zorder=4,
             label="window position (u)")
    ax2.plot(t, df["f"], drawstyle="steps-post", color=INK_MUTED, lw=1.0,
             ls=(0, (2, 1.5)), zorder=3, label="fan speed (f)")
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1], ["off / shut", "on / open"])
    ax2.set_xlabel("minutes")
    ax2.grid(False)

    legend_below(axes[1], ncol=5, gap=32, source=[axes[0], axes[1]])
    extra = [caption(axes[1], note, gap=52)] if note else None
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.18, bbox_extra_artists=extra)
        plt.close(fig)
        print(f"  wrote {path}")
        return None
    return fig, axes
