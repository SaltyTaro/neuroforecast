"""Figures for the technical report's two sealed tests, drawn from the shipped per-case evidence tables.

Writes docs/figures/{design_timeline,external_per_date,external_risk_coverage,external_by_state,assay_decision}.png.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import neuroforecast_gate as gate  # noqa: E402

FIG = ROOT / "docs/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
V2, PERSIST, V1 = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8d8c86"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, dtype={"identity": str, "spid": str})


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.primary.astype(bool) & frame.usable.astype(bool)].reset_index(drop=True)


def to_datetime(dates) -> pd.Series:
    return pd.to_datetime(pd.Series(dates).astype(str), format="%Y%m%d")


def design_timeline() -> None:
    dev = read("evaluation/v2_prepared/dev_cases.csv")
    ext = read("evaluation/v2_prepared/ext_cases.csv")
    reserve = [20170920, 20171004, 20171011]
    v1_dev = sorted(set(dev.date) - set(reserve) - {20170201, 20170222})
    rows = [("External test, 24 batches", sorted(set(ext.date)), V2, "scored once, 27 Sep 2026"),
            ("v1 development, 12", v1_dev, NEUTRAL, ""),
            ("v1 reserve, 3", reserve, PERSIST, "scored once, 21 Sep 2026"),
            ("unscored in v1, 2", [20170201, 20170222], "#b9b8b1", "")]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for y, (label, dates, color, note) in enumerate(reversed(rows)):
        x = to_datetime(dates)
        ax.scatter(x, np.full(len(x), y), s=34, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if note:
            ax.text(x.max() + pd.Timedelta(days=25), y, note, va="center", color=INK2, fontsize=7.5)
    ax.axvspan(to_datetime([20160701])[0], to_datetime([20171031])[0], color="#f1f0ec", zorder=0)
    ax.text(to_datetime([20160715])[0], 3.55, "all 17 dates here were v2 development", color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlim(to_datetime([20140101])[0], to_datetime([20181231])[0])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_title("Two sealed tests, from different periods")
    fig.tight_layout()
    fig.savefig(FIG / "design_timeline.png", dpi=200)
    plt.close(fig)


def per_date() -> None:
    a, b = primary(read("evaluation/external/arm_a_cases.csv")), primary(read("evaluation/external/arm_b_cases.csv"))
    loss = {"v2": gate.date_losses(b), "persistence": gate.date_losses(b.assign(prediction=b.day7)), "v1": gate.date_losses(a)}
    table = pd.DataFrame(loss).sort_values("persistence")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, (_, row) in enumerate(table.iterrows()):
        ax.plot([row.v2, row.persistence], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter(table.persistence, y, s=40, color=PERSIST, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Carry day 7 forward")
    ax.scatter(table.v1, y, s=26, marker="D", color=V1, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Frozen v1 tool (two inputs missing)")
    ax.scatter(table.v2, y, s=40, color=V2, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="v2 forecast")
    labels = [f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}" for d in table.index]
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("Mean absolute error, log2 control-relative network spikes (lower is better)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("External test: v2 beat carrying day 7 forward on 19 of 24 batches")
    means = {k: v.mean() for k, v in loss.items()}
    ax.text(0.99, 0.02, f"date-macro MAE  v2 {means['v2']:.3f}   persistence {means['persistence']:.3f}   frozen v1 {means['v1']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=7.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "external_per_date.png", dpi=200)
    plt.close(fig)


def risk_coverage() -> None:
    b = primary(read("evaluation/external/arm_b_cases.csv"))
    error = abs(b.prediction - b.target).to_numpy()
    error_p = abs(b.day7 - b.target).to_numpy()
    retention = np.arange(50, 101)

    def curve(score, err):
        order = np.argsort(score, kind="stable")
        return [err[order][: int(np.ceil(r / 100 * len(err)))].mean() for r in retention]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(retention, curve(b.trust_score.to_numpy(), error), color=V2, linewidth=2, label="v2 forecast, declining by difficulty (locked)")
    ax.plot(retention, curve(abs(b.prediction.to_numpy()), error), color=V2, linewidth=2, linestyle=(0, (4, 3)), label="v2 forecast, declining the most extreme")
    ax.plot(retention, curve(b.trust_score.to_numpy(), error_p), color=PERSIST, linewidth=2, label="Carry day 7 forward, same difficulty verdict")
    ax.axhline(error.mean(), color=NEUTRAL, linewidth=1, linestyle=":", label="v2 forecast, random retention")
    kept = 100 * (~b.declined.astype(bool)).mean()
    ax.axvline(kept, color=GRID, linewidth=1)
    ax.text(kept - 0.6, ax.get_ylim()[1] * 0.98, f"locked threshold keeps {kept:.0f}%", ha="right", va="top", color=INK2, fontsize=7.5)
    ax.set_xlabel("Conditions kept (%)")
    ax.set_ylabel("MAE of kept conditions")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("External test: the difficulty verdict helps both predictors")
    fig.tight_layout()
    fig.savefig(FIG / "external_risk_coverage.png", dpi=200)
    plt.close(fig)


def by_state() -> None:
    b = primary(read("evaluation/external/arm_b_cases.csv"))
    states = [("strongly\nsuppressed\n≤ −1", b.day7 <= -1), ("suppressed\n−1 to −0.5", (b.day7 > -1) & (b.day7 <= -0.5)),
              ("near control\nwithin ±0.5", abs(b.day7) < 0.5), ("raised\n0.5 to 1", (b.day7 >= 0.5) & (b.day7 < 1)),
              ("hyperactive\n≥ 1", b.day7 >= 1)]
    x = np.arange(len(states))
    v2 = [abs(b.prediction - b.target)[m].mean() for _, m in states]
    carry = [abs(b.day7 - b.target)[m].mean() for _, m in states]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    width = 0.36
    for offset, values, color, label in [(-width / 2, v2, V2, "v2 forecast"), (width / 2, carry, PERSIST, "Carry day 7 forward")]:
        ax.bar(x + offset, values, width - 0.04, color=color, label=label, zorder=3)
        for xi, v in zip(x + offset, values):
            ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.set_ylim(0, max(v2 + carry) * 1.15)
    ax.set_xticks(x, [f"{label}\n{int(m.sum())} conditions" for label, m in states], fontsize=7.5)
    ax.set_ylabel("MAE, log2 (lower is better)")
    ax.set_xlabel("Day-7 state, log2 relative to the plate's controls")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    ax.set_title("External test, post hoc: the forecast wins where day 7 is far from control")
    fig.tight_layout()
    fig.savefig(FIG / "external_by_state.png", dpi=200)
    plt.close(fig)


def assay_decision() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.35))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.8, 1.0)
    ax.axis("off")
    ax.plot([0.5, 13.5], [0, 0], color=GRID, linewidth=3, solid_capstyle="round")
    for day, label, color in [(0.7, "plating", NEUTRAL), (5, "DIV 5\nrecord", INK2), (7, "DIV 7\nrecord + decide", V2), (9, "DIV 9\nrecord", INK2), (12, "DIV 12\nendpoint", PERSIST)]:
        ax.scatter([day], [0], s=90 if day == 7 else 50, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(day, 0.35, label, ha="center", va="bottom", color=INK if day == 7 else INK2, fontsize=8, fontweight="bold" if day == 7 else "normal")
    ax.text(7, -0.45, "At day 7 the tool answers:  is this batch's reference usable?  what will day 12 be?  can that forecast be trusted?",
            ha="center", va="top", color=INK2, fontsize=7.8)
    ax.annotate("", xy=(12, -0.15), xytext=(7, -0.15), arrowprops={"arrowstyle": "->", "color": V2, "linewidth": 1.2})
    fig.tight_layout()
    fig.savefig(FIG / "assay_decision.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    assay_decision()
    design_timeline()
    per_date()
    risk_coverage()
    by_state()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
