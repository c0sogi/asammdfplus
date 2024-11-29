from inspect import signature
from os import PathLike
from typing import (
    Final,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    ParamSpec,
    Sequence,
    TypeAlias,
    TypeVar,
)

import numpy as np
from asammdf import Signal

CompressionSuffix: TypeAlias = Literal[
    ".gz",
    ".bz2",
    ".zip",
    ".xz",
    ".zst",
    ".tar",
    ".tar.gz",
    ".tar.xz",
    ".tar.bz2",
]

LineStyle: TypeAlias = Literal[
    "-",
    "--",
    "-.",
    ":",
    "None",
    " ",
    "",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]

ColorLike: TypeAlias = str | tuple[float, float, float, float]
PathOrPaths: TypeAlias = str | PathLike | Iterable[str | PathLike]
Groups: TypeAlias = (
    Mapping[str, Sequence[str]] | Sequence[str | Sequence[str]] | str
)


SIG_SIGNS: Final = signature(Signal.__init__).parameters
EXCLUDED_COLUMNS: Final = ("samples", "timestamps")


SelfType = TypeVar("SelfType")
ReturnType = TypeVar("ReturnType")
P = ParamSpec("P")
V = TypeVar("V")

default_dtype = np.float64


MDF_SUFFIXES = (".mf4", ".mf3", ".mdf", ".dat")
_quotes: str = r"['\"]"
_non_quotes: str = r"[^'\"]"
MDF_ACCESS_PATTERN = (
    "mdf" + r"\[" + _quotes + rf"({_non_quotes}+)" + _quotes + r"\]"
)
del _quotes, _non_quotes


class Package(NamedTuple):
    name: str
    timestamps: np.ndarray
    samples: np.ndarray
    label: str
    color: tuple[float, float, float, float]


class GroupProperty(NamedTuple):
    same_range: bool
    tickless: bool
    signals: Sequence[str]
