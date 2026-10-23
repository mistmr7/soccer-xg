"""Side-by-side pitch heatmaps: shot volume (log scale) and conversion
rate (hexes with a minimum shot count), attacking half only.

    uv run python -m soccerxg.pitch_maps
    uv run python -m soccerxg.pitch_maps --min-shots 50 --gridsize 40

Writes docs/figs/shot_maps.png by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Arc

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "docs" / "figs"

LINE = "#555555"


def draw_half_pitch(ax) -> None:
    """StatsBomb coordinates, attacking left to right, goal at x=120."""
    ax.plot([60, 60], [0, 80], color=LINE, lw=1.2)
    ax.plot([60, 120], [0, 0], color=LINE, lw=1.2)
    ax.plot([60, 120], [80, 80], color=LINE, lw=1.2)
    ax.plot([120, 120], [0, 80], color=LINE, lw=1.6)

    ax.plot([102, 102], [18, 62], color=LINE, lw=1.2)
    ax.plot([102, 120], [18, 18], color=LINE, lw=1.2)
    ax.plot([102, 120], [62, 62], color=LINE, lw=1.2)

    ax.plot([114, 114], [30, 50], color=LINE, lw=1.2)
    ax.plot([114, 120], [30, 30], color=LINE, lw=1.2)
    ax.plot([114, 120], [50, 50], color=LINE, lw=1.2)

    ax.plot([120, 120], [36, 44], color=LINE, lw=3.5)

    ax.plot(108, 40, marker="o", color=LINE, ms=2.5)
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232,
                     color=LINE, lw=1.2))
    ax.add_patch(Arc((60, 40), 20, 20, angle=0, theta1=-52, theta2=52,
                     color=LINE, lw=1.2))

    ax.set_xlim(58, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.set_xlabel("distance along the pitch (StatsBomb x)")
    ax.set_ylabel("width (StatsBomb y)")


def plot_maps(df: pd.DataFrame, out: Path, gridsize: int = 36,
              min_shots: int = 30) -> None:
    d = df[df.x >= 60].copy()
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    hb0 = axes[0].hexbin(d.x, d.y, gridsize=gridsize, cmap="Blues",
                         norm=LogNorm(vmin=1), mincnt=1)
    draw_half_pitch(axes[0])
    axes[0].set_title("Shot volume (log color scale)")
    fig.colorbar(hb0, ax=axes[0], label="shots", shrink=0.8)

    hb1 = axes[1].hexbin(d.x, d.y, C=d.goal.astype(float), gridsize=gridsize,
                         cmap="Blues", reduce_C_function=np.mean,
                         mincnt=min_shots, vmin=0.0, vmax=0.6)
    draw_half_pitch(axes[1])
    axes[1].set_title(f"Conversion rate (hexes with {min_shots}+ shots)")
    fig.colorbar(hb1, ax=axes[1], label="goal rate", shrink=0.8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} ({len(d):,} shots)")


WOMENS = {"FA Women's Super League", "UEFA Women's Euro", "Women's World Cup"}


def main() -> None:
    from .features import build_shots

    ap = argparse.ArgumentParser()
    ap.add_argument("--gridsize", type=int, default=36)
    ap.add_argument("--min-shots", type=int, default=30)
    ap.add_argument("--population", choices=["all", "men", "women"],
                    default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = build_shots()
    if args.population == "men":
        df = df[~df.competition.isin(WOMENS)]
    elif args.population == "women":
        df = df[df.competition.isin(WOMENS)]
    out = Path(args.out) if args.out else FIGS / f"shot_maps_{args.population}.png"
    plot_maps(df, out, args.gridsize, args.min_shots)


if __name__ == "__main__":
    main()
