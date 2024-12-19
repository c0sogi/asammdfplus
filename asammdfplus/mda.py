import logging
from inspect import signature
from itertools import cycle
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    cast,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ._original import MDF, Signal
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


class PlotSettings(TypedDict, total=False):
    vars: list[str]
    ylim: tuple[float, float]
    show_legend: bool


class CSTPlotConfig(TypedDict):
    engine_speed_col: str
    batt_vol_col: str
    raster: float
    fig_size: tuple[float, float]
    show_plot: bool
    grid: bool
    xlim: Optional[tuple[float, float]]
    legend_location: str
    start_offset: float
    mdf_dict: dict[str, MDF | str]
    color_map: dict[str, str]
    plots_config: dict[str, PlotSettings]
    signal_time_options: Optional[
        dict[str, list[tuple[Literal["relative", "absolute"], float]]]
    ]


def find_engine_starting_points(
    df_or_mdf: MDF | pd.DataFrame,
    engine_speed_col: str,
    batt_vol_col: str,
    debounce_patience: float = 0.5,
    start_patience: float = 5.0,
    battery_drop_patience: float = 1.0,
    engine_lower_threshold: float = 250.0,
    engine_upper_threshold: float = 1000.0,
    predicate: Callable[[float, float], bool] | None = None,
    visualize: bool = False,
) -> list[tuple[float, float]]:
    """Find the engine starting points based on the given conditions."""
    if isinstance(df_or_mdf, pd.DataFrame):
        batt_vol: pd.Series = df_or_mdf[batt_vol_col]
        engine_speed: pd.Series = df_or_mdf[engine_speed_col]
    else:
        batt_vol_signal: Signal = df_or_mdf.get(batt_vol_col)
        engine_speed_signal: Signal = df_or_mdf.get(engine_speed_col)

        batt_vol: pd.Series = pd.Series(
            batt_vol_signal.samples,
            index=batt_vol_signal.timestamps,
            name=batt_vol_col,
        )
        engine_speed: pd.Series = pd.Series(
            engine_speed_signal.samples,
            index=engine_speed_signal.timestamps,
            name=engine_speed_col,
        )

    batt_vol_diff: pd.Series = batt_vol.diff(1)
    is_low_speed: pd.Series = engine_speed < engine_lower_threshold
    is_high_speed: pd.Series = engine_speed > engine_upper_threshold

    # Calculate the indices where the engine speed crosses the low speed threshold
    low_edges: list[float] = [
        float(f)
        for f in engine_speed[
            ~is_low_speed
            & is_low_speed.shift(1, fill_value=is_low_speed.iloc[0])
        ].index.to_list()
    ]
    high_edges: list[float] = [
        float(f)
        for f in engine_speed[
            is_high_speed
            & ~is_high_speed.shift(1, fill_value=is_high_speed.iloc[0])
        ].index.to_list()
    ]

    # Implement a debouncing mechanism
    filtered_ref_edges: list[float] = []
    previous_edge: float | None = None
    for edge in low_edges:
        if (
            previous_edge is None
            or (edge - previous_edge) > debounce_patience
        ):
            filtered_ref_edges.append(edge)
            previous_edge = edge

    # Find the actual start points based on battery voltage drop
    actual_start_points: list[float] = []
    for ref_edge in filtered_ref_edges:
        start_index: float = max(0, ref_edge - battery_drop_patience)
        end_index: float = ref_edge
        if predicate is not None and not predicate(start_index, end_index):
            continue
        search_window: pd.Series = batt_vol_diff[
            (batt_vol_diff.index >= start_index)
            & (batt_vol_diff.index <= end_index)
        ]
        if not search_window.empty:
            min_point = float(search_window.idxmin())
            actual_start_points.append(min_point)

    timestamps: list[tuple[float, float]] = []
    min_high_edge: float = float("inf")
    closest_high_edge: float | None
    for start_point in reversed(actual_start_points):
        closest_high_edge = None
        for high_edge in high_edges:
            if (
                high_edge > start_point
                and high_edge <= start_point + start_patience
            ):
                if high_edge < min_high_edge:
                    closest_high_edge = high_edge
                break

        if closest_high_edge is not None:
            min_high_edge = min(min_high_edge, closest_high_edge)
            timestamps.append((start_point, closest_high_edge))
    timestamps = sorted(timestamps, key=lambda x: x[0])

    # Plot vertical lines at success and fail end points
    if visualize:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = cast(np.ndarray, ax)
        ax[0].plot(engine_speed, label="Engine Speed", color="blue")
        ax[1].plot(batt_vol, label="Battery Voltage", color="orange")
        ax[2].plot(
            batt_vol_diff, label="Battery Voltage Diff", color="darkorange"
        )
        ax[0].vlines(
            [start for start, end in timestamps],
            ymin=engine_speed.min(),
            ymax=engine_speed.max(),
            color="green",
            linestyle="--",
        )
        ax[0].vlines(
            [end for start, end in timestamps],
            ymin=engine_speed.min(),
            ymax=engine_speed.max(),
            color="red",
            linestyle="--",
        )
        for a in ax:
            a.legend()
        plt.show()
    return timestamps


def plot_cst(cst_plot_config: CSTPlotConfig) -> tuple[Figure, list[Axes]]:

    engine_speed_col = cst_plot_config["engine_speed_col"]
    batt_vol_col = cst_plot_config["batt_vol_col"]
    raster = cst_plot_config["raster"]
    fig_size = cst_plot_config["fig_size"]
    show_plot = cst_plot_config["show_plot"]
    global_grid = cst_plot_config["grid"]
    global_xlim = cst_plot_config["xlim"]
    legend_location = cst_plot_config["legend_location"]
    start_offset = cst_plot_config["start_offset"]

    mdf_dict = cst_plot_config["mdf_dict"]
    color_map_config = cst_plot_config["color_map"]
    plots_config = cst_plot_config["plots_config"]
    signal_time_options = cst_plot_config["signal_time_options"]

    # 필요한 변수 수집
    needed_vars: set[str] = set()
    for plot_config in plots_config.values():
        for var in plot_config.get("vars", []):
            needed_vars.add(var)

    df_dict: dict[str, pd.DataFrame] = {}
    start_times: dict[str, float] = {}
    file_names: dict[str, str] = {}

    # MDF -> DataFrame 변환 및 엔진 스타트 포인트 계산
    for name, mdf in mdf_dict.items():
        if not isinstance(mdf, MDF):
            mdf = MDF(mdf)
        existing_vars = [v for v in needed_vars if v in mdf]
        if not existing_vars:
            # 필요한 변수가 하나도 없는 경우 빈 DataFrame
            df_original = pd.DataFrame()
        else:
            df_original = mdf.to_dataframe(existing_vars, raster=raster)

        df_original = df_original.sort_index()

        start_points: list[tuple[float, float]] = (
            find_engine_starting_points(
                df_or_mdf=mdf,
                engine_speed_col=engine_speed_col,
                batt_vol_col=batt_vol_col,
                visualize=False,
            )
        )
        if start_points:
            start_time, end_time = start_points[0]
        else:
            start_time = (
                df_original.index.min() if not df_original.empty else 0.0
            )

        df = df_original.copy()
        df.index = df.index - start_time + start_offset

        df_dict[name] = df
        start_times[name] = start_time

        f_name = (
            Path(mdf.name).stem if hasattr(mdf, "name") else f"{name} Data"
        )
        file_names[name] = f_name

    # 바이너리 플롯 판별
    binary_plots: dict[str, bool] = {}
    for plot_title, plot_config in plots_config.items():
        vars_list = plot_config.get("vars", [])
        # data_combined를 리스트로 관리 후 empty 아닌 것만 concat
        non_empty_series: list[pd.Series] = []
        for source_name, df in df_dict.items():
            for var in vars_list:
                if var in df.columns:
                    s = df[var].dropna()
                    if not s.empty:
                        non_empty_series.append(s)

        if non_empty_series:
            data_combined = pd.concat(non_empty_series)
            unique_vals = data_combined.unique()
            if set(unique_vals).issubset({0, 1}):
                binary_plots[plot_title] = True
            else:
                binary_plots[plot_title] = False
        else:
            binary_plots[plot_title] = False

    # 높이 비율 결정
    height_ratios = []
    for plot_title in plots_config:
        if binary_plots.get(plot_title, False):
            height_ratios.append(1)
        else:
            height_ratios.append(3)

    # figure 생성 시 constrained_layout 사용
    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    gs = GridSpec(
        len(plots_config), 1, height_ratios=height_ratios, hspace=0.5
    )
    axes: list[Axes] = []
    shared_ax = None

    for i in range(len(plots_config)):
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            shared_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=shared_ax)
        axes.append(ax)

    if global_grid:
        for ax in axes:
            ax.grid(True)

    if global_xlim and shared_ax is not None:
        shared_ax.set_xlim(global_xlim)

    # 색상 매핑
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    given_color_map = color_map_config.copy()
    for i, name in enumerate(mdf_dict.keys()):
        if name not in given_color_map:
            given_color_map[name] = colors[i % len(colors)]
    final_color_map = given_color_map

    # Plot 그리기
    for i, (plot_title, plot_config) in enumerate(plots_config.items()):
        vars_list = plot_config.get("vars", [])
        ylim = plot_config.get("ylim", None)
        show_legend = plot_config.get("show_legend", True)

        axes[i].set_title(plot_title)
        data_plotted = False

        # 각 plot에 대해 vars 리스트에서 첫 번째로 유효한 신호 하나만 선택해서 그리기
        for name, df in df_dict.items():
            # 순서대로 신호 탐색
            selected_var: Optional[str] = None
            for var_candidate in vars_list:
                if (
                    var_candidate in df.columns
                    and not (
                        numeric_series := df[var_candidate].dropna()
                    ).empty
                    and pd.api.types.is_numeric_dtype(numeric_series)
                ):
                    selected_var = var_candidate
                    break

            if selected_var:
                df[selected_var].plot.line(
                    ax=axes[i],
                    label=f"{name}: {selected_var}",
                    color=final_color_map[name],
                    ylim=ylim,  # type: ignore
                    legend=False,
                )
                data_plotted = True

                # 바이너리 플롯인 경우 fill_between 처리
                if binary_plots.get(plot_title, False):
                    axes[i].set_ylim(-0.2, 1.2)
                    axes[i].fill_between(
                        df.index,
                        0,
                        df[selected_var],
                        step="pre",
                        alpha=0.3,
                        color=final_color_map[name],
                    )

        if not data_plotted:
            axes[i].text(
                0.5,
                0.5,
                "No Signal",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[i].transAxes,
                fontsize="small",
                color="red",
            )
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        if ylim and not binary_plots.get(plot_title, False):
            axes[i].set_ylim(ylim)

        if show_legend and data_plotted:
            legend = axes[i].legend(loc=legend_location)
            if legend:
                legend.get_frame().set_facecolor("none")

        # signal_time_options 처리
        if signal_time_options:
            for source_name, options in signal_time_options.items():
                if source_name not in df_dict:
                    # Unknown source, skip
                    continue

                source_df = df_dict[source_name]
                # 여기서도 plot할 때 사용한 var를 찾아야 함
                # 마찬가지로 vars_list 순서대로 탐색
                var_for_annot = None
                for var_candidate in vars_list:
                    if var_candidate not in source_df.columns:
                        continue

                    numeric_series = pd.to_numeric(
                        source_df[var_candidate], errors="coerce"
                    )
                    if (~numeric_series.isna()).any():
                        source_df[var_candidate] = (
                            numeric_series.ffill().bfill()
                        )
                        var_for_annot = var_candidate
                        break

                if not var_for_annot:
                    continue

                start_time_original = start_times[source_name]
                color = final_color_map[source_name]

                for mode, time_value in options:
                    if mode == "relative":
                        target_time = time_value
                    elif mode == "absolute":
                        target_time = (
                            time_value - start_time_original + start_offset
                        )
                    else:
                        # Unknown mode, skip
                        continue

                    if (
                        not source_df.empty
                        and var_for_annot in source_df.columns
                        and not source_df[var_for_annot].dropna().empty
                    ):
                        try:
                            time_diff = np.abs(source_df.index - target_time)
                            nearest_idx = time_diff.argmin()
                            actual_time = source_df.index[int(nearest_idx)]
                        except Exception:
                            continue

                        value = source_df.at[actual_time, var_for_annot]
                        ylow, yhigh = axes[i].get_ylim()

                        display_value = value
                        va = "bottom"
                        if value > yhigh:
                            display_value = yhigh
                            va = "top"
                        elif value < ylow:
                            display_value = ylow
                            va = "bottom"

                        ha = "left"
                        axes[i].annotate(
                            f"{value:.2f}",
                            xy=(actual_time, display_value),
                            xytext=(actual_time, display_value),
                            textcoords="data",
                            fontsize="small",
                            color=color,
                            verticalalignment=va,
                            horizontalalignment=ha,
                        )

                        axes[i].axvline(
                            x=actual_time,
                            color=color,
                            linestyle="--",
                            alpha=0.7,
                        )

    if axes:
        axes[-1].set_xlabel("Time (s) from Engine Start + Offset")

    file_info_str = "\n".join(
        [f"{name}: {fname}" for name, fname in file_names.items()]
    )
    fig.suptitle(file_info_str, y=0.93)

    if show_plot:
        # constrained_layout=True를 사용했으므로 tight_layout은 사용하지 않음
        # plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    return fig, axes


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
                samples=np.asarray(signal.samples),
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
            where=(package.samples > 0).ravel().tolist(),
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
                np.asarray(signal_dict[name].samples).min()
                for name in group_property.signals
            )
            max_value = max(
                np.asarray(signal_dict[name].samples).max()
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
                    where=(package.samples > 0).ravel().tolist(),
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
