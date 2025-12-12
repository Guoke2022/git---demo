"""
Combined figure:
Row 1: error-bar style panels (mean ± 2SD, with box-like glyph)
Row 2: three line charts
Row 3: three heatmaps + shared vertical colorbar

Refactored for GitHub:
- English comments
- Encapsulated functions
- Centralized configuration
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# ------------------------------ Configuration ------------------------------ #

@dataclass(frozen=True)
class PathsConfig:
    """File paths configuration."""
    high_csv: Path
    low_csv: Path
    mid_csv: Path
    overall_csv: Path
    heat_source_csv: Path
    output_png: Path


@dataclass(frozen=True)
class PlotConfig:
    """Plot styling & scaling configuration."""
    # scaling
    x_scale: float = 100.0        # option_distance_100_km -> km
    value_scale: float = 100.0    # WTT scale multiplier

    # heatmap grouping
    group_interval: float = 1.0
    max_home_road_dist: float = 81.0

    # x ticks for heatmap
    heat_xtick_step: int = 10
    heat_xmax_tick: int = 80

    # figure size
    fig_size: tuple[int, int] = (20, 16)
    dpi: int = 600

    # row layout
    height_ratios: tuple[float, float, float] = (0.6, 1.0, 1.0)
    left: float = 0.05
    right: float = 0.97
    top: float = 0.96
    bottom: float = 0.10
    wspace: float = 0.25
    hspace: float = 0.35

    # third row position adjustments
    shift_inner: float = 0.04
    shift_row_right: float = 0.02


def set_global_style() -> None:
    """Apply global matplotlib + seaborn style (match your original look)."""
    warnings.filterwarnings("ignore")

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.titlesize": 30,
        "font.family": "Arial",

        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    sns.set_style("white")  # white background, no grid


def build_colormap() -> LinearSegmentedColormap:
    """
    Soft colormap. (Kept consistent with your current setting)
    If you want true "blue-white-red", add a blue color at the beginning.
    """
    return LinearSegmentedColormap.from_list(
        "rb_soft",
        [
            "#ffffff",  # center white
            "#545454",  
            "#2F2F2F",  
        ],
        N=256
    )


# ------------------------------ Data Preparation ------------------------------ #

def reorder_to_categories(
    means: np.ndarray,
    sds: np.ndarray,
    order: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder arrays to match categories order."""
    return means[order], sds[order]


def load_line_series(paths: PathsConfig, cfg: PlotConfig) -> dict[str, np.ndarray]:
    """Load line-chart CSVs and return scaled x and y series."""
    high_df = pd.read_csv(paths.high_csv)
    low_df = pd.read_csv(paths.low_csv)
    mid_df = pd.read_csv(paths.mid_csv)
    all_df = pd.read_csv(paths.overall_csv)

    x = high_df.iloc[:, 1].to_numpy() * cfg.x_scale

    def col(df: pd.DataFrame, idx: int) -> np.ndarray:
        return df.iloc[:, idx].to_numpy() * cfg.value_scale

    series = {
        "x": x,

        "grade_high": col(high_df, 2),
        "grade_low": col(low_df, 2),
        "grade_mid": col(mid_df, 2),
        "grade_all": col(all_df, 2),

        "beds_high": col(high_df, 3),
        "beds_low": col(low_df, 3),
        "beds_mid": col(mid_df, 3),
        "beds_all": col(all_df, 3),

        "rep_high": col(high_df, 4),
        "rep_low": col(low_df, 4),
        "rep_mid": col(mid_df, 4),
        "rep_all": col(all_df, 4),
    }
    return series


def compute_dist_grade_ratio_from_csv(
    csv_path: Path,
    group_interval: float,
    max_home_road_dist: float,
) -> pd.DataFrame:
    """
    Group by distance bins and compute within-(grade, SES) proportions.

    Expected columns:
    - home_road_dist
    - grade
    - SES_group
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Trim long tail
    df = df[df["home_road_dist"] < max_home_road_dist].copy()

    # Distance binning
    df["dist_group"] = (df["home_road_dist"] / group_interval).astype(int) * group_interval
    df["dist_group"] = df["dist_group"].round(2)

    grouped = (
        df.groupby(["grade", "dist_group", "SES_group"])
        .size()
        .reset_index(name="count")
    )

    total_per_grade = (
        grouped.groupby(["grade", "SES_group"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )

    grouped = grouped.merge(total_per_grade, on=["grade", "SES_group"], how="left")
    grouped["ratio"] = grouped["count"] / grouped["total"]
    return grouped


def build_heat_datasets(grouped: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Build heatmap datasets for Overall / High-SES / Low-SES."""
    return [
        ("Overall", grouped.copy()),
        ("High-SES", grouped[grouped["SES_group"] == "High-SES"].copy()),
        ("Low-SES", grouped[grouped["SES_group"] == "Low-SES"].copy()),
    ]


# ------------------------------ Plot Helpers ------------------------------ #

def plot_error_panel(
    ax: plt.Axes,
    categories: list[str],
    x_pos: np.ndarray,
    means: np.ndarray,
    errs: np.ndarray,
    title: str,
    color_map: dict[str, str],
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    """
    Draw a box-like glyph around mean with whiskers for mean ± 2SD (errs = 2*SD).
    """
    box_width = 0.35
    cap_width = box_width * 0.45

    data_min = float(np.min(means - errs))
    data_max = float(np.max(means + errs))
    span = (data_max - data_min) if data_max > data_min else 1.0
    star_offset = 0.02 * span

    for i, (m, e) in enumerate(zip(means, errs)):
        color = color_map[categories[i]]
        sd = e / 2.0
        x_ctr = x_pos[i]

        # q1/q3 for visual box (approx: +/-0.674 SD around mean)
        q1 = m - 0.674 * sd
        q3 = m + 0.674 * sd

        x0 = x_ctr - box_width / 2.0
        rect = plt.Rectangle(
            (x0, q1),
            box_width,
            q3 - q1,
            edgecolor="black",
            facecolor=color,
            linewidth=1.1,
            zorder=3
        )
        ax.add_patch(rect)

        # mean line (as "median" line)
        ax.plot([x0, x0 + box_width], [m, m], color="black", linewidth=1.2, zorder=4)

        # whiskers for mean ± 2SD
        y_low = m - e
        y_high = m + e

        ax.vlines(x_ctr, y_low, q1, colors="black", linewidth=1.2, zorder=4)
        ax.vlines(x_ctr, q3, y_high, colors="black", linewidth=1.2, zorder=4)

        ax.hlines(y_low, x_ctr - cap_width / 2, x_ctr + cap_width / 2,
                  colors="black", linewidth=1.2, zorder=4)
        ax.hlines(y_high, x_ctr - cap_width / 2, x_ctr + cap_width / 2,
                  colors="black", linewidth=1.2, zorder=4)

        # significance stars (kept as original)
        ax.text(x_ctr, y_high + star_offset, "***", ha="center", va="bottom",
                fontsize=22, color="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)

    # bottom-right panel title
    ax.text(1.02, 0.02, title, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=22, fontweight="bold", color="black")

    if ymin is None:
        ymin = data_min - 0.05 * (data_max - data_min)
    if ymax is None:
        ymax = data_max + 0.05 * (data_max - data_min)
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_line_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y_high: np.ndarray,
    y_mid: np.ndarray,
    y_low: np.ndarray,
    y_all: np.ndarray,
    xlabel: str,
    ylabel: str | None,
    ylim: tuple[float | None, float | None],
    colors: dict[str, str],
    show_legend: bool = False,
) -> None:
    """Draw one line chart panel with shaded region after overall peak."""
    ax.plot(
        x, y_high, marker="o", linestyle="--", linewidth=3.0, markersize=9,
        color=colors["high"], label="High",
        markeredgecolor="black", markeredgewidth=1
    )
    ax.plot(
        x, y_low, marker="s", linestyle="-", linewidth=3.0, markersize=9,
        color=colors["low"], label="Low",
        markeredgecolor="black", markeredgewidth=1
    )
    ax.plot(
        x, y_mid, marker="s", linestyle="--", linewidth=3.0, markersize=9,
        color=colors["mid"], label="Mid",
        markeredgecolor="black", markeredgewidth=1
    )
    ax.plot(
        x, y_all, marker="o", linestyle="-", linewidth=3.0, markersize=9,
        color=colors["all"], label="Overall",
        markeredgecolor="black", markeredgewidth=1
    )

    ax.set_title("")  # keep blank (same as your original)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # ylim
    y_min, y_max = ylim
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = float(np.max([y_high.max(), y_mid.max(), y_low.max(), y_all.max()]) * 1.1)
    ax.set_ylim(y_min, y_max)

    # peak shading based on overall
    x_peak = x[int(np.argmax(y_all))]
    right_edge = ax.get_xlim()[1]

    ax.axvline(x_peak, color="#8F8F8F", linewidth=2, linestyle=(0, (10, 6)), zorder=-1)
    ax.axvspan(x_peak, right_edge, facecolor="0.8", alpha=0.4, zorder=-1)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()

        # Desired legend order: Low, High, Mid, Overall (match your code behavior)
        desired = ["Low", "High", "Mid", "Overall"]
        idx = [labels.index(name) for name in desired]

        ax.legend(
            [handles[i] for i in idx], [labels[i] for i in idx],
            loc="upper right",
            bbox_to_anchor=(0.98, 1.0),
            frameon=False,
            ncol=2,
            handlelength=0.9,
            handletextpad=0.4,
            columnspacing=0.5,
            labelspacing=0.2,
            borderaxespad=0.25,
            borderpad=0.2,
            markerscale=0.9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_heat_panel(
    ax: plt.Axes,
    title: str,
    data: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    vmax: float,
    grade_map: dict[str, int],
    xticks: np.ndarray,
    show_ylabel: bool,
    show_arrows: bool,
) -> None:
    """Draw one heatmap panel."""
    heat = (
        data.pivot_table(index="grade", columns="dist_group", values="ratio", fill_value=0)
        .sort_index()
    )

    heat.index = heat.index.map(grade_map)
    heat = heat.sort_index(ascending=False)

    sns.heatmap(
        heat,
        cmap=cmap,
        linewidths=0,
        vmin=0,
        vmax=vmax,
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Travel cost (km)")
    ax.set_ylabel("Hospital quality (grade)" if show_ylabel else "")

    # arrows (kept consistent with your original placement logic)
    if show_arrows:
        ax.annotate(
            "", xy=(1.05, -0.12), xytext=(-0.085, -0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=4, color="black")
        )
        ax.annotate(
            "", xy=(-0.08, 1), xytext=(-0.08, -0.125),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=4, color="black")
        )
    else:
        ax.annotate(
            "", xy=(1.05, -0.12), xytext=(0, -0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=4, color="black")
        )

    ax.set_yticklabels(heat.index, rotation=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # bottom-right title
    ax.text(
        0.98, 0.02, title,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=22, fontweight="bold", color="black"
    )


def add_panel_labels(all_axes: list[plt.Axes]) -> None:
    """Add a, b, c... labels to panels (skip last two as in original)."""
    labels = list("abcdefghij")  # 10 panels

    for idx, (ax, lab) in enumerate(zip(all_axes, labels)):
        if idx in (8, 9):  # skip i, j
            continue
        x_pos = -0.08
        if idx == 7:  # panel 'h' special shift (your original tweak)
            x_pos = -0.15
        ax.text(
            x_pos, 1.07, lab,
            transform=ax.transAxes,
            fontsize=28, fontweight="bold",
            va="top", ha="left",
            color="black"
        )


# ------------------------------ Main Plot Assembly ------------------------------ #

def make_figure(
    cfg: PlotConfig,
    paths: PathsConfig,
) -> None:
    """Build the combined figure and save it."""
    set_global_style()
    cmap = build_colormap()

    # ---- Error-bar row data (kept as your fixed arrays) ----
    categories = ["Low", "Mid", "High", "Overall"]
    x_cat = np.arange(len(categories))

    cat_colors = {
        "Overall": "#f7ae55",
        "High": "#92B5C9",
        "Mid": "#9AD175",
        "Low": "#E59091",
    }

    # Original order in arrays: [Overall, High, Mid, Low]
    grade_mean = np.array([0.5869351555, 0.3190434681, 0.5240850936, 0.9233726850])
    grade_sd   = np.array([0.0006851517, 0.0000245951, 0.0019170548, 0.0021384037])

    beds_mean = np.array([0.0388085702, 0.0091799104, 0.0411664216, 0.0285616280])
    beds_sd   = np.array([0.0000194020, 0.0010841538, 0.0002515575, 0.0002804927])

    rep_mean = np.array([0.0083636822, 0.0091799104, 0.0083696870, 0.0074545083])
    rep_sd   = np.array([0.0072154526, 0.0025987759, 0.0049392441, 0.0147018812])

    distance_mean = np.array([-34.4186961308, -59.1666475308, -40.5206043849, -27.3759121900])
    distance_sd   = np.array([7.7676736899, 11.3457807458, 7.4384694959, 7.4414089396])

    # Reorder to match categories: Low, Mid, High, Overall
    order = [3, 2, 1, 0]
    grade_mean, grade_sd = reorder_to_categories(grade_mean, grade_sd, order)
    beds_mean, beds_sd = reorder_to_categories(beds_mean, beds_sd, order)
    rep_mean, rep_sd = reorder_to_categories(rep_mean, rep_sd, order)
    distance_mean, distance_sd = reorder_to_categories(distance_mean, distance_sd, order)

    grade_err = 2 * grade_sd
    beds_err = 2 * beds_sd
    rep_err = 2 * rep_sd
    distance_err = 2 * distance_sd

    # ---- Line chart data ----
    series = load_line_series(paths, cfg)

    # ---- Heatmap data ----
    grouped = compute_dist_grade_ratio_from_csv(
        csv_path=paths.heat_source_csv,
        group_interval=cfg.group_interval,
        max_home_road_dist=cfg.max_home_road_dist
    )
    heat_datasets = build_heat_datasets(grouped)
    global_vmax = float(grouped["ratio"].max())

    grade_map = {
        "Primary": 1,
        "Secondary": 2,
        "Tertiary": 3,
        "Tertiary grade A": 4,
    }

    # ---- Figure layout ----
    fig = plt.figure(figsize=cfg.fig_size)

    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=cfg.height_ratios,
        left=cfg.left, right=cfg.right, top=cfg.top, bottom=cfg.bottom,
        wspace=cfg.wspace, hspace=cfg.hspace
    )

    # Row 0: error panels (1x4)
    sub_gs_err = gs[0, :].subgridspec(1, 4, wspace=0.25)
    axes_err = [fig.add_subplot(sub_gs_err[0, j]) for j in range(4)]

    plot_error_panel(axes_err[0], categories, x_cat, grade_mean, grade_err, "Grade", cat_colors)
    plot_error_panel(axes_err[1], categories, x_cat, beds_mean, beds_err, "Capacity", cat_colors,
                     ymin=0.0005, ymax=0.05)
    plot_error_panel(axes_err[2], categories, x_cat, rep_mean, rep_err, "Reputation", cat_colors)
    plot_error_panel(axes_err[3], categories, x_cat, distance_mean, distance_err, "Distance", cat_colors)

    axes_err[0].set_ylabel("Coefficient", fontsize=26)

    # Row 1: 3 line panels
    axes_line = [fig.add_subplot(gs[1, j]) for j in range(3)]

    line_colors = {
        "high": "#92B5C9",
        "low":  "#E59091",
        "mid":  "#9AD175",
        "all":  "#f7ae55",
    }

    # y-limits (kept from your original)
    ylim_grade = (0, None)
    ylim_beds = (0, 0.65)
    ylim_rep = (0, 0.28)

    plot_line_panel(
        axes_line[0], series["x"],
        series["grade_high"], series["grade_mid"], series["grade_low"], series["grade_all"],
        xlabel="Option distance (km)",
        ylabel="WTT(km) by grade",
        ylim=ylim_grade,
        colors=line_colors
    )
    plot_line_panel(
        axes_line[1], series["x"],
        series["beds_high"], series["beds_mid"], series["beds_low"], series["beds_all"],
        xlabel="Option distance (km)",
        ylabel="WTT(km) by capacity",
        ylim=ylim_beds,
        colors=line_colors
    )
    plot_line_panel(
        axes_line[2], series["x"],
        series["rep_high"], series["rep_mid"], series["rep_low"], series["rep_all"],
        xlabel="Option distance (km)",
        ylabel="WTT(km) by reputation",
        ylim=ylim_rep,
        colors=line_colors,
        show_legend=True
    )

    # Row 2: 3 heat panels
    axes_heat = [fig.add_subplot(gs[2, j]) for j in range(3)]

    xticks = np.arange(0, cfg.heat_xmax_tick + 1, cfg.heat_xtick_step)

    for i, (ax, (title, data)) in enumerate(zip(axes_heat, heat_datasets)):
        plot_heat_panel(
            ax=ax,
            title=title,
            data=data,
            cmap=cmap,
            vmax=global_vmax,
            grade_map=grade_map,
            xticks=xticks,
            show_ylabel=(i == 0),
            show_arrows=(i == 0),
        )

    # Adjust heatmap row spacing/position (match your original tweak)
    for idx, ax in enumerate(axes_heat):
        pos = ax.get_position()
        if idx > 0:
            pos.x0 -= cfg.shift_inner * idx
            pos.x1 -= cfg.shift_inner * idx
        pos.x0 += cfg.shift_row_right
        pos.x1 += cfg.shift_row_right
        ax.set_position(pos)

    # Shared colorbar for heatmaps
    norm = plt.Normalize(0, global_vmax)
    pos = axes_heat[-1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.018, pos.height])

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical"
    )
    cbar.set_label("Proportion (%)", fontsize=24)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(labelsize=22)

    # Thicken spines for row 0 and row 1 panels
    for ax in axes_err + axes_line:
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

    # Panel labels
    all_axes = axes_err + axes_line + axes_heat
    add_panel_labels(all_axes)

    # Shift y-label positions (keep your original layout intent)
    axes_err[0].yaxis.set_label_coords(-0.15, 0.5)
    axes_line[0].yaxis.set_label_coords(-0.10, 0.5)
    axes_heat[0].yaxis.set_label_coords(-0.175, 0.5)

    # Move heatmap x-label down a bit
    for ax in axes_heat:
        ax.xaxis.set_label_coords(0.5, -0.15)

    # Save & show
    paths.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(paths.output_png, dpi=cfg.dpi, bbox_inches="tight")
    plt.show()


# ------------------------------ Entrypoint ------------------------------ #

def main() -> None:
    # TODO: Update these paths to your local/project layout.
    DATA_DIR = Path(r"./")
    paths = PathsConfig(
        high_csv=DATA_DIR / "hight.csv",
        low_csv=DATA_DIR / "low.csv",
        mid_csv=DATA_DIR / "middle.csv",
        overall_csv=DATA_DIR / "result.csv",
        heat_source_csv=Path("./part.csv"),
        output_png=Path("./outputs/figure5.png"),
    )

    cfg = PlotConfig()
    make_figure(cfg, paths)


if __name__ == "__main__":
    main()

