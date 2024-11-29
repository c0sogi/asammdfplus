import logging
from inspect import signature
from itertools import cycle
from typing import Callable, Iterator, Mapping, Sequence, TypeAlias

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from asammdf import MDF, Signal
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._typing import (
    ColorLike,
    GroupProperty,
    Groups,
    LineStyle,
    Package,
)
from ._utils import CaselessDict

SignalDict: TypeAlias = CaselessDict[Signal]
logger = logging.getLogger(__name__)


try:
    from matplotlib import colormaps
except ImportError:
    import matplotlib.cm as colormaps


def plot(
    mdf_or_df: MDF | pd.DataFrame,
    groups: Groups,
    fig_cols: int = 1,
    figsize_per_row: tuple[float, float] = (10, 2),
    dpi: int | None = None,
    ylims: Mapping[str, tuple[float, float]] | None = None,
    tickless_groups: Sequence[str] | str | None = None,
    timestamps_list: list[tuple[float, float]] | None = None,
    cmap: str = "tab10",
    title_format: str | None = None,
    hide_ax_title: bool = True,
    bit_signal_ratio: float = 0.2,
    bit_signal_alpha: float = 0.5,
    bit_signal_ylim: tuple[float, float] = (-0.2, 1.2),
    spine_offset: float = 50,
    legend_loc: str = "upper right",
    line_styles: dict[str, LineStyle] | LineStyle | None = None,
    colors: dict[str, ColorLike] | None = None,
    markers: dict[str, str] | str | None = None,
    markersize: int | None = None,
    grid: bool = False,
) -> list[Figure]:
    """Plot multiple signals in a single figure.

    Args:
        mdf_or_df: MDF file or DataFrame.
        groups: Group of signals to plot.
        fig_cols: Number of columns in the figure.
        figsize_per_row: Size of the figure per row.
        dpi: Dots per inch.
        ylims: Y-axis limits.
        tickless_groups: Group names to hide the y-axis ticks.
        timestamps_list: List of timestamps to plot.
        cmap: Colormap.
        title_format: Title format.
        hide_ax_title: Whether to hide the axis title.
        bit_signal_ratio: Ratio of the bit signal height.
        bit_signal_alpha: Alpha value of the bit signal.
        bit_signal_ylim: Y-axis limits of the bit signal.
        spine_offset: Offset of the y-axis spine.
        legend_loc: Location of the legend.
        line_styles: Line styles.
        colors: Colors of the signals.
        markers: Markers of the signals.
        markersize: Marker size.
        grid: Whether to show the grid.

    Returns:
        list[Figure]: List of figures."""
    group_properties: Mapping[str, GroupProperty] = _make_group_properties(
        groups,
        (
            ()
            if tickless_groups is None
            else (
                [tickless_groups]
                if isinstance(tickless_groups, str)
                else tickless_groups
            )
        ),
    )
    del groups

    get_signal: Callable[[str], Signal] = _get_signal(mdf_or_df)
    signal_dict: SignalDict = SignalDict(
        {
            signal: get_signal(signal)
            for group_property in group_properties.values()
            for signal in group_property.signals
        }
    )
    del get_signal

    bit_groups: Sequence[bool] = tuple(
        all(
            signal_dict[signal_name].bit_count == 1
            for signal_name in group_property.signals
        )
        for group_property in group_properties.values()
    )
    ylims = _make_ylims(signal_dict, group_properties, ylims)

    colormap = (
        colormaps.get_cmap(  # pyright: ignore[reportAttributeAccessIssue]
            cmap
        )
    )
    num_plots: int = len(group_properties)
    num_rows = int(-1 * (num_plots / fig_cols) // 1 * -1)
    figs: list[Figure] = []
    for fig_idx, timestamps in enumerate(
        timestamps_list or _create_empty_timestamps_list(signal_dict)
    ):
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=fig_cols,
            sharex=True,
            height_ratios=[
                bit_signal_ratio if is_bit else 1 for is_bit in bit_groups
            ][:num_rows],
            figsize=(
                int(figsize_per_row[0]),
                int(figsize_per_row[1] * num_rows),
            ),
            dpi=dpi,
        )
        axs_flattened: Sequence[Axes] = axes.flatten() if num_plots > 1 else [axes]  # type: ignore
        color_cycle = cycle(
            [colormap(i) for i in range(getattr(colormap, "N", 1))]
        )
        for ax_idx, (group_name, group_property) in enumerate(
            group_properties.items()
        ):
            # Initialize plotting data structures
            packages: list[Package] = _prep_packages(
                group_property.signals,
                signal_dict,
                color_cycle,
                timestamps,
                colors,
            )

            # Plot on the same axis with multiple y-axes
            ax: Axes = _plot_ax(
                ax=axs_flattened[ax_idx],
                packages=packages,
                group_name=group_name,
                line_styles=line_styles,
                markers=markers,
                markersize=markersize,
                ylims=ylims,
                bit_signal_alpha=bit_signal_alpha,
                bit_signal_ylim=bit_signal_ylim,
                spine_offset=spine_offset,
                is_all_bit_signal=bit_groups[ax_idx],
                legend_loc=legend_loc,
                hide_ax_title=hide_ax_title,
                tickless=group_property.tickless,
                grid=grid,
            )

            # If no signal is plotted, put `No data` in the middle of the subplot
            if not packages:
                start, end = timestamps
                ax.text(
                    (start + end) / 2,
                    0.5,
                    f"No data for {group_name}",
                    ha="center",
                    va="center",
                    color="yellow",
                    fontsize=20,
                    weight="bold",
                    bbox=dict(
                        facecolor="red",
                        alpha=0.5,
                        edgecolor="red",
                        boxstyle="round,pad=1",
                    ),
                )

        # Set x-axis label for the last row
        for lower_ax in axs_flattened[-fig_cols:]:
            lower_ax.set_xlabel("Time [s]")

        # Set title for the first row
        if title_format is not None:
            fig.suptitle(
                _format_title_string(
                    title_string_format=title_format,
                    idx=fig_idx,
                    timestamps=timestamps,
                )
            )

        # Tighten the layout and append the figure to the list
        fig.tight_layout()
        figs.append(fig)

    return figs


def _format_title_string(
    title_string_format: str, idx: int, timestamps: tuple[float, float]
) -> str:
    """Format the title string."""
    start, end = timestamps
    return title_string_format.format(
        idx=idx,
        n=idx + 1,
        start=start,
        end=end,
    )


def _prep_packages(
    signal_names: Sequence[str],
    all_signals: SignalDict,
    color_cycle: Iterator[tuple[float, float, float, float]],
    timestamps: tuple[float, float],
    custom_colors: dict[str, ColorLike] | None,
) -> list[Package]:
    packages: list[Package] = []
    start, end = timestamps
    for signal_name in signal_names:
        signal = all_signals[signal_name].cut(start, end)
        if custom_colors is not None and signal_name in custom_colors:
            color = custom_colors[signal_name]
        else:
            color = next(color_cycle)
        packages.append(
            Package(
                name=signal_name,
                timestamps=signal.timestamps,
                samples=signal.samples,
                label=f"{signal_name}\n({str(signal.unit)})",
                color=mcolors.to_rgba(color),
            )
        )
    return packages


def _create_empty_timestamps_list(
    signal_dict: SignalDict,
) -> list[tuple[float, float]]:
    min_start: float = float("inf")
    for name in signal_dict:
        timestamps = signal_dict[name].timestamps
        if timestamps.size > 0:
            min_start = min(min_start, timestamps[0])
    max_end: float = float("-inf")
    for name in signal_dict:
        timestamps = signal_dict[name].timestamps
        if timestamps.size > 0:
            max_end = max(max_end, timestamps[-1])
    return [(min_start, max_end)]


def _plot_signal(
    ax: Axes,
    package: Package,
    line_styles: dict[str, LineStyle] | LineStyle | None,
    markers: dict[str, str] | str | None,
    markersize: int | None,
) -> None:
    ax.plot(
        package.timestamps,
        package.samples,
        color=package.color,
        label=package.name,
        linestyle=(
            line_styles
            if isinstance(line_styles, str)
            else (
                line_styles.get(package.name, "-")
                if isinstance(line_styles, dict)
                else "-"
            )
        ),
        marker=(
            markers
            if isinstance(markers, str)
            else (
                markers.get(package.name, None)
                if isinstance(markers, dict)
                else None
            )
        ),
        markersize=markersize,
    )


def _set_twinax_properties(
    ax: Axes,
    package: Package,
    signal_name: str,
    ylims: Mapping[str, tuple[float, float]],
    bit_signal_alpha: float,
    is_bit_signal: bool,
    legend_loc: str,
    spine_offset: float | None,
):
    ax.set_ylabel(package.label, color=package.color)
    ax.tick_params(axis="y", labelcolor=package.color)

    if spine_offset is not None:
        ax.spines["right"].set_position(("outward", spine_offset))
        ax.spines["right"].set_color(package.color)

    if signal_name in ylims:
        ax.set_ylim(*ylims[signal_name])

    if is_bit_signal:
        ax.fill_between(
            package.timestamps,
            0,
            package.samples,
            where=(package.samples > 0).tolist(),
            color=package.color,
            alpha=bit_signal_alpha,
        )


def _make_group_properties(
    groups: Groups, tickless_groups: Sequence[str]
) -> dict[str, GroupProperty]:
    def group_generator() -> Iterator[tuple[str, str | Sequence[str]]]:
        if isinstance(groups, Mapping):
            for group_name, single_or_group in groups.items():
                yield group_name, single_or_group
        else:
            for group_name, single_or_group in enumerate(
                (groups,) if isinstance(groups, str) else groups, 1
            ):
                yield f"GROUP{group_name}", single_or_group

    single_or_group: str | Sequence[str]
    group_properties: dict[str, GroupProperty] = {}

    for group_name, single_or_group in group_generator():
        signals: Sequence[str]
        is_same_range: bool = False
        if isinstance(single_or_group, str):
            signals = (single_or_group,)
        else:
            signals = single_or_group
            if isinstance(single_or_group, tuple):
                is_same_range = True

        group_properties[group_name] = GroupProperty(
            same_range=is_same_range,
            signals=signals,
            tickless=group_name in tickless_groups,
        )

    return group_properties


def _make_ylims(
    signal_dict: SignalDict,
    group_properties: dict[str, GroupProperty],
    ylims: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    if ylims is None:
        ylims = {}
    else:
        ylims = {**ylims}  # To avoid modifying the original ylims

    for group_property in group_properties.values():
        if not group_property.same_range:
            continue
        existing_min_max: tuple[float, float] | None = next(
            (
                ylims[signal_name]
                for signal_name in group_property.signals
                if signal_name in ylims
            ),
            None,
        )
        if existing_min_max is not None:
            # If the same range group is defined, use the same range for all signals in the group
            for signal_name in group_property.signals:
                ylims[signal_name] = existing_min_max
        else:
            # If the same range group is not defined, use the min and max of all signals in the group
            min_value = min(
                signal_dict[name].samples.min()
                for name in group_property.signals
            )
            max_value = max(
                signal_dict[name].samples.max()
                for name in group_property.signals
            )
            for signal_name in group_property.signals:
                ylims[signal_name] = (min_value, max_value)
    return ylims


def _get_signal(mdf_or_df: MDF | pd.DataFrame) -> Callable[[str], Signal]:
    if isinstance(mdf_or_df, pd.DataFrame):

        def get_signal(name: str) -> Signal:
            signatures = signature(Signal.__init__).parameters
            return Signal(
                samples=np.asarray(mdf_or_df[name].values),
                timestamps=np.asarray(mdf_or_df.index.values),
                **{
                    k: v
                    for k, v in mdf_or_df.attrs.get(
                        name, {"name": name}
                    ).items()
                    if k in signatures
                },
            )

        return get_signal
    else:
        return mdf_or_df.get


def _plot_ax(
    ax: Axes,
    packages: Sequence[Package],
    group_name: str,
    line_styles: dict[str, LineStyle] | LineStyle | None,
    markers: dict[str, str] | str | None,
    markersize: int | None,
    ylims: Mapping[str, tuple[float, float]],
    bit_signal_alpha: float,
    bit_signal_ylim: tuple[float, float],
    spine_offset: float,
    is_all_bit_signal: bool,
    legend_loc: str,
    hide_ax_title: bool,
    tickless: bool,
    grid: bool,
) -> Axes:
    offsets: float = 0.0
    is_main_ax: bool = True
    for package in packages:
        if is_main_ax or tickless:
            ax.grid(grid)
            _plot_signal(ax, package, line_styles, markers, markersize)

            if tickless:
                ax.legend(loc=legend_loc, fontsize="small")
                ax.set_ylabel("")
                ax.set_yticks([])
            else:
                ax.set_ylabel(package.label, color=package.color)
                ax.tick_params(axis="y", labelcolor=package.color)

            if not hide_ax_title:
                ax.set_title(group_name)
            if package.name in ylims:
                ax.set_ylim(*ylims[package.name])
            if is_all_bit_signal:
                ax.set_ylabel("")
                ax.legend(loc=legend_loc, fontsize="small")
                ax.set_ylim(*bit_signal_ylim)
                ax.fill_between(
                    package.timestamps,
                    0,
                    package.samples,
                    where=(package.samples > 0).tolist(),
                    color=package.color,
                    alpha=bit_signal_alpha,
                )
            is_main_ax = False
        else:
            ax_new: Axes = (
                ax.twinx()
            )  # pyright: ignore[reportAssignmentType]
            ax_new.grid(grid)
            _plot_signal(ax_new, package, line_styles, markers, markersize)
            _set_twinax_properties(
                ax_new,
                package,
                package.name,
                ylims,
                bit_signal_alpha,
                is_all_bit_signal,
                legend_loc,
                offsets,
            )
            offsets += spine_offset
    return ax
