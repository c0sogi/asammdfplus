import logging
import re
from contextlib import contextmanager
from functools import partial, wraps
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    TypeGuard,
)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ._original import MDF, ChannelsType, Signal
from ._typing import (
    EXCLUDED_COLUMNS,
    SIG_SIGNS,
    P,
    PathOrPaths,
    ReturnType,
    SelfType,
    V,
)

if TYPE_CHECKING:
    from .mdf import MDFPlus


logger = logging.getLogger(__name__)


class CaselessDict(dict[str, V]):
    """A case-insensitive dictionary"""

    def __call__(self, key: str) -> V:
        return self[key]

    def __getitem__(self, key: str) -> V:
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(k.lower() == key.lower() for k in self.keys())

    def __or__(self, other: dict[str, V]) -> "CaselessDict[V]":  # type: ignore
        return CaselessDict({**self, **other})

    def __ior__(self, other: dict[str, V]) -> "CaselessDict[V]":  # type: ignore
        self.update(other)
        return self

    def __and__(self, other: dict[str, V]) -> "CaselessDict[V]":
        return CaselessDict({k: self[k] for k in self if k in other})

    def __iand__(self, other: dict[str, V]) -> "CaselessDict[V]":
        self = CaselessDict({k: self[k] for k in self if k in other})
        return self

    def get(self, key: str, default: V | None = None) -> V | None:  # type: ignore
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default

    def get_with_wildcard(self, key_pattern: str) -> V | None:
        pattern = re.compile(key_pattern.lower().replace("*", ".*"))
        return next(
            (v for k, v in self.items() if pattern.fullmatch(k)), None
        )

    def get_with_regex(self, regex_pattern: str) -> V | None:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
        return next(
            (v for k, v in self.items() if pattern.fullmatch(k)), None
        )


class Counter(dict[str, int]):
    """A dictionary that returns 0 for missing keys and increments the value for existing keys."""

    def __missing__(self, key: str) -> int:
        """Return 0 for missing keys."""
        return 0

    def add(self, key: str) -> int:
        """Increment the value for the key and return the new value."""
        self[key] += 1
        return self[key]

    @classmethod
    def make_unique_strings(cls, strings: Iterable[str]) -> list[str]:
        """Ensure that the strings are unique by appending a suffix to duplicates."""
        self = cls()
        return [
            string if self.add(string) == 1 else f"{string}_{self[string]}"
            for string in strings
        ]


def get_channel_names_with_device(*name_sources: Iterable[str]) -> set[str]:
    """Get the MDF channel names, in favor of the device name seperated by `\\`."""
    registry: dict[str, set[str]] = {}
    for name_source in name_sources:
        for string in name_source:
            if "\\" in string:
                name, device = string.split("\\", 1)
                if name not in registry:
                    registry[name] = set()
                registry[name].add(device)
            else:
                if string not in registry:
                    registry[string] = set()
    return {
        f"{name}\\{device}" if device else name
        for name, devices in registry.items()
        for device in (devices or (None,))
    }


def get_channel_names_without_device(
    *name_sources: Iterable[str],
) -> set[str]:
    """Get the MDF channel names, without the device name seperated by `\\`."""
    return set(
        name.split("\\")[0]
        for name_source in name_sources
        for name in name_source
    )


def hijack_channels(
    mdfplus: "MDFPlus", channels: ChannelsType
) -> tuple[dict[str, Signal], ChannelsType]:
    hijacked_channels: list[str] = []
    non_hijacked_channels: ChannelsType = []
    for channel in channels:
        if channel in mdfplus.__cache__:
            hijacked_channels.append(channel)
        else:
            non_hijacked_channels.append(
                channel  # pyright: ignore[reportArgumentType]
            )
    hijacked_signals: dict[str, Signal] = {
        name: mdfplus.__cache__[name] for name in hijacked_channels
    }
    return hijacked_signals, non_hijacked_channels


def proxy_function_as_method(
    func: Callable[
        [SelfType], Callable[Concatenate[SelfType, P], ReturnType]
    ]
):
    """A decorator to proxy the property to the method.

    If you want to use extend a function from other module as method, you can use this decorator.
    The method must be wrapped with `@property` decorator before using this decorator.
    Automatically, the first argument of the method is the instance itself.
    Also, IDEs can recognize the input and return types of the method, so it is useful for type hinting.
    """

    @wraps(func)
    def wrapper(self) -> Callable[P, ReturnType]:
        return partial(func(self), self)

    return wrapper


def concat_dataframes_with_sequence(
    dfs: Sequence[pd.DataFrame],
    axis: Literal[0, 1] = 0,
    sorted_columns: bool = True,
) -> pd.DataFrame:
    """Concatenate DataFrames with the same columns and index names, ensuring a continuous sequence in index with a hop inferred from the previous DataFrame."""
    # Variable to track the starting index for the next DataFrame
    next_start_index: float = 0
    # Initialize a list to store adjusted DataFrames
    adjusted_dfs: list[pd.DataFrame] = []

    for i, df in enumerate(dfs):
        if i > 0:
            df = df.set_index(df.index + next_start_index)

        # Update next_start_index for the next DataFrame
        if len(df.index) > 1:
            next_start_index = df.index[-1] + (
                dfs[i - 1].index[-1] - dfs[i - 1].index[-2]
            )
        adjusted_dfs.append(df)

    # Concatenate all adjusted DataFrames
    result: pd.DataFrame = pd.concat(adjusted_dfs, axis=axis)
    for df in dfs:
        result.attrs.update(df.attrs)
    return (
        result.reindex(sorted(result.columns), axis=1)
        if sorted_columns
        else result
    )


def get_filepaths(
    path_or_paths: PathOrPaths,
    ext_or_exts: str | Iterable[str] = (".dat", ".mf4"),
) -> list[Path]:
    """Parse file paths from a path or paths.

    Args:
        path_or_paths (PathOrPaths): A path or paths.
        exts (Iterable[str], optional): A list of file extensions. Defaults to (".dat", ".mf4").
    """
    file_paths: list[Path] = []
    if isinstance(ext_or_exts, str):
        ext_or_exts = (ext_or_exts,)
    else:
        ext_or_exts = tuple(ext.lower() for ext in ext_or_exts)

    def parse_file(path: Path) -> None:
        if Path(path).is_file() and Path(path).suffix.lower() in ext_or_exts:
            file_paths.append(path)
        elif Path(path).is_dir():
            file_paths.extend(
                p
                for ext in ext_or_exts
                for p in Path(path).rglob(f"*{ext}", case_sensitive=False)
            )

    if isinstance(path_or_paths, (str, PathLike)):
        parse_file(Path(path_or_paths))
    elif isinstance(path_or_paths, Iterable):
        for path in path_or_paths:
            if isinstance(path, (str, PathLike)):
                parse_file(Path(path))
            else:
                logger.warning(f"Invalid path: {path}")
    else:
        logger.warning(f"Invalid path_or_paths: {path_or_paths}")

    if not file_paths:
        logger.warning(f"No files found: {path_or_paths}")
    return file_paths


def is_asammdf_object(obj: object) -> bool:
    """Check if the object is an ASAMMDF object."""
    return (
        True
        if getattr(obj, "__module__", "").startswith("asammdf.")
        else False
    )


def is_MDFPlus_instance(mdf: "MDF | MDFPlus") -> TypeGuard["MDFPlus"]:
    """Check if the MDF object is an MDFPlus object."""
    return isinstance(mdf, MDF) and hasattr(mdf, "__cache__")


def make_pattern(*names: str) -> re.Pattern[str]:
    return re.compile(
        "|".join(
            (
                "^" + re.escape(name).replace("\\*", ".*") + "$"
                for name in names
            )
        )
    )


def filter_names(
    names: set[str],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[str]:
    if include:
        pattern = make_pattern(*include)
        return [name for name in names if pattern.match(name)]
    elif exclude:
        pattern = make_pattern(*exclude)
        return [name for name in names if not pattern.match(name)]
    return list(names)


def df_factory(
    mdf: "MDF | MDFPlus",
    raster: float | None,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    reduce_memory_usage: bool,
    allow_asammdf_objects: bool,
) -> pd.DataFrame:
    is_mdf_plus: bool = is_MDFPlus_instance(mdf)
    if include or exclude:
        names: set[str]
        if is_mdf_plus:
            names = mdf.channel_names
        else:
            names = set(mdf.channels_db.keys())
        channels = filter_names(names, include, exclude)
    else:
        channels = None
    metadata: dict[Hashable, dict[str, Any]] = {
        signal.name.split("\\")[0]: {
            k: v
            for k, v in signal.__dict__.items()
            if k in SIG_SIGNS
            and k not in EXCLUDED_COLUMNS
            and (allow_asammdf_objects or not is_asammdf_object(v))
        }
        for signal in (
            mdf.iter_channels() if channels is None else mdf.select(channels)
        )
    }
    df: pd.DataFrame = mdf.to_dataframe(
        channels, reduce_memory_usage=reduce_memory_usage, raster=raster
    )
    if is_mdf_plus:
        timestamps: np.ndarray = df.index.values
        for name, signal in mdf.__cache__.items():
            df[name] = mdf.signal_to_series(signal.interp(timestamps))
    df.columns = Counter.make_unique_strings(
        (str(col).split("\\")[0] for col in df.columns)
    )
    # Update the attributes of the DataFrame
    df.attrs.update(metadata)
    return df


@contextmanager
def matplotlib_params(
    figure_figsize: tuple[float, float] | None = None,
    lines_linewidth: float | None = None,
    lines_color: str | None = None,
    lines_linestyle: str | None = None,
    axes_labelsize: str | None = None,
    axes_titlesize: str | None = None,
    axes_titleweight: str | None = None,
    font_size: float | None = None,
    font_family: str | None = None,
    grid: bool | None = None,
    legend_loc: str | None = None,
    legend_fontsize: str | None = None,
    savefig_dpi: float | None = None,
    savefig_format: str | None = None,
    xtick_labelsize: str | None = None,
    ytick_labelsize: str | None = None,
):
    """Set multiple matplotlib parameters temporarily."""
    params: dict[str, Any] = {
        "figure.figsize": figure_figsize,
        "lines.linewidth": lines_linewidth,
        "lines.color": lines_color,
        "lines.linestyle": lines_linestyle,
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "axes.titleweight": axes_titleweight,
        "font.size": font_size,
        "font.family": font_family,
        "axes.grid": grid,
        "legend.loc": legend_loc,
        "legend.fontsize": legend_fontsize,
        "savefig.dpi": savefig_dpi,
        "savefig.format": savefig_format,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
    }

    # Filter out None values
    params = {k: v for k, v in params.items() if v is not None}

    original_params = {key: plt.rcParams[key] for key in params.keys()}
    try:
        plt.rcParams.update(params)
        yield
    finally:
        plt.rcParams.update(original_params)
