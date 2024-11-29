import logging
import re
from inspect import signature
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    TypeGuard,
)

import numpy as np
import pandas as pd
from asammdf import MDF, Signal

from ..pandas.dataframe import Counter, concat_dataframes_with_sequence
from ..pathlib.utils import get_filepaths
from ..typings.pathlib import PathOrPaths

if TYPE_CHECKING:
    from .mdf import MDFPlus

SIG_SIGNS: Final = signature(Signal.__init__).parameters
EXCLUDED_COLUMNS: Final = ("samples", "timestamps")


logger = logging.getLogger(__name__)


def _is_asammdf_object(obj: object) -> bool:
    """Check if the object is an ASAMMDF object."""
    return (
        True
        if getattr(obj, "__module__", "").startswith("asammdf.")
        else False
    )


def _is_MDFPlus_instance(mdf: "MDF | MDFPlus") -> TypeGuard["MDFPlus"]:
    """Check if the MDF object is an MDFPlus object."""
    return isinstance(mdf, MDF) and hasattr(mdf, "__cache__")


def _make_pattern(*names: str) -> re.Pattern[str]:
    return re.compile(
        "|".join(
            (
                "^" + re.escape(name).replace("\\*", ".*") + "$"
                for name in names
            )
        )
    )


def _filter_names(
    names: set[str],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[str]:
    if include:
        pattern = _make_pattern(*include)
        return [name for name in names if pattern.match(name)]
    elif exclude:
        pattern = _make_pattern(*exclude)
        return [name for name in names if not pattern.match(name)]
    return list(names)


def _df_factory(
    mdf: "MDF | MDFPlus",
    raster: float | None,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    reduce_memory_usage: bool,
    allow_asammdf_objects: bool,
) -> pd.DataFrame:
    is_mdf_plus: bool = _is_MDFPlus_instance(mdf)
    channels: list[str] | None = (
        _filter_names(
            (
                mdf.channel_names
                if is_mdf_plus
                else set(mdf.channels_db.keys())
            ),
            include,
            exclude,
        )
        if include or exclude
        else None
    )
    metadata: dict[Hashable, dict[str, Any]] = {
        signal.name.split("\\")[0]: {
            k: v
            for k, v in signal.__dict__.items()
            if k in SIG_SIGNS
            and k not in EXCLUDED_COLUMNS
            and (allow_asammdf_objects or not _is_asammdf_object(v))
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
            df.loc[:, name] = mdf.signal_to_series(signal.interp(timestamps))
    df.columns = Counter.make_unique_strings(
        (str(col).split("\\")[0] for col in df.columns)
    )
    # Update the attributes of the DataFrame
    df.attrs.update(metadata)
    return df


def convert_mdf_to_dataframe(
    fpath_or_mdf: str | Path | MDF,
    raster: float | None = 0.1,
    compression_suffix: str = ".xz",
    reduce_memory_usage: bool = True,
    ignore_existing: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    write_to_disk: bool = True,
    allow_asammdf_objects: bool = False,
    **kwargs: Any,  # For compatibility
) -> pd.DataFrame:
    """Load MDF, resample, ensure unique column names, and save as compressed DataFrame.

    Args:
        fpath: Path to the MDF file.
        raster: Sampling interval for resampling.
        compression_suffix: File extension for compressed saving.
        reduce_memory_usage: Whether to downcast the DataFrame to reduce memory usage.
        ignore_existing: Whether to ignore existing compressed files.
        include: List of signal names to include.
        exclude: List of signal names to exclude.
        write_to_disk: Whether to save the DataFrame to disk.
        allow_asammdf_objects: Whether to include ASAMMDF objects in attributes of the DataFrame.
        prefer_original_df_conversion: Whether to use the original DataFrame factory.

    Returns:
        The path to the saved DataFrame.
    """
    # == Initialize ==
    if include and isinstance(include, str):
        include = (include,)
    elif exclude and isinstance(exclude, str):
        exclude = (exclude,)

    if isinstance(fpath_or_mdf, MDF):
        # MDF 객체가 주어진 경우, 파일 경로를 추출
        fpath = Path(fpath_or_mdf.name)
    else:
        # 파일 경로가 주어진 경우, Path 객체로 변환
        fpath = Path(fpath_or_mdf)

    # 압축 파일 확장자 추가
    out_fpath: Path = fpath.with_suffix(compression_suffix)

    # Ignore existing 조건이 거짓이면서 이미 파일이 존재하는 경우, 파일을 불러와서 반환
    if not ignore_existing and out_fpath.exists():
        logger.info(f"- {fpath_or_mdf} already exists, skipping")
        return pd.read_pickle(out_fpath)

    if isinstance(fpath_or_mdf, MDF):
        mdf = fpath_or_mdf
    else:
        mdf = MDF(fpath, raise_on_multiple_occurrences=False)

    df: pd.DataFrame = _df_factory(
        mdf=mdf,
        raster=raster,
        include=include,
        exclude=exclude,
        reduce_memory_usage=reduce_memory_usage,
        allow_asammdf_objects=allow_asammdf_objects,
    )
    df.attrs.update(
        {
            "$raster": raster,
            "$compression_suffix": compression_suffix,
            "$reduce_memory_usage": reduce_memory_usage,
            "$fpath": fpath.as_posix(),
            "$last_call_info": mdf.last_call_info,
            "$start_time": mdf.start_time,
        }
    )

    del mdf  # 메모리 해제
    if write_to_disk:
        df.to_pickle(out_fpath)
        logger.info(
            f"- Saved to {out_fpath} ({df.shape[0]} rows, {df.shape[1]} columns)"
        )
    return df


def convert_mdfs_to_dataframe(
    path_or_paths: PathOrPaths,
    ext_or_exts: str | Iterable[str] = (".dat", ".mf4"),
    raster: float | None = 0.1,
    compression_suffix: str = ".xz",
    reduce_memory_usage: bool = True,
    ignore_existing: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    write_to_disk: bool = True,
    allow_asammdf_objects: bool = False,
    axis: Literal[0, 1] = 0,
    sorted_columns: bool = True,
    prefer_original_df_conversion: bool | None = None,
) -> pd.DataFrame:
    """Convert MDF files to DataFrames and concatenate them along the specified axis.

    Args:
        path_or_paths (PathOrPaths): A path or paths.
        ext_or_exts (str | Iterable[str], optional): A list of file extensions. Defaults to (".dat", ".mf4").
        raster (float | None, optional): Sampling interval for resampling. Defaults to 0.1.
        compression_suffix (str, optional): File extension for compressed saving. Defaults to ".xz".
        reduce_memory_usage (bool, optional): Whether to downcast the DataFrame to reduce memory usage. Defaults to True.
        ignore_existing (bool, optional): Whether to ignore existing compressed files. Defaults to False.
        include (Sequence[str] | None, optional): List of signal names to include. Defaults to None.
        exclude (Sequence[str] | None, optional): List of signal names to exclude. Defaults to None.
        write_to_disk (bool, optional): Whether to save the DataFrame to disk. Defaults to True.
        allow_asammdf_objects (bool, optional): Whether to include ASAMMDF objects in attributes of the DataFrame. Defaults to False.
        axis (Literal[0, 1], optional): The axis to concatenate the DataFrames. Defaults to 0.
        sorted_columns (bool, optional): Whether to sort the columns of the concatenated DataFrame. Defaults to True.
        prefer_original_df_conversion (bool | None, optional): Whether to use the original DataFrame factory. Defaults to None.

    Returns:
        pd.DataFrame: A concatenated DataFrame.
    """
    return concat_dataframes_with_sequence(
        dfs=[
            convert_mdf_to_dataframe(
                fpath_or_mdf=file_path,
                raster=raster,
                compression_suffix=compression_suffix,
                reduce_memory_usage=reduce_memory_usage,
                ignore_existing=ignore_existing,
                include=include,
                exclude=exclude,
                write_to_disk=write_to_disk,
                allow_asammdf_objects=allow_asammdf_objects,
                prefer_original_df_conversion=prefer_original_df_conversion,
            )
            for file_path in get_filepaths(
                path_or_paths=path_or_paths, ext_or_exts=ext_or_exts
            )
        ],
        axis=axis,
        sorted_columns=sorted_columns,
    )


def read_mdf_from_dataframe(fpath_or_df: str | Path | pd.DataFrame) -> MDF:
    """Load DataFrame, ensure unique column names, and save as MDF.

    Args:
        fpath: Path to the DataFrame file.

    Returns:
        The MDF object.
    """
    if isinstance(fpath_or_df, pd.DataFrame):
        df = fpath_or_df
    else:
        df: pd.DataFrame = pd.read_pickle(fpath_or_df)

    mdf = MDF()
    if isinstance(fpath_or_df, (str, Path)):
        mdf.name = Path(fpath_or_df).stem

    if df.attrs:
        mdf.append(
            [
                Signal(
                    samples=df[k],
                    timestamps=df.index,
                    **{k: v for k, v in v.items() if k in SIG_SIGNS},
                )
                for k, v in df.attrs.items()
            ]
        )
    else:
        logger.warning(
            "No attributes found in the DataFrame. Using default values."
        )
        mdf.append(
            [
                Signal(samples=df[k], timestamps=df.index, name=k)
                for k in df.columns
            ]
        )
    return mdf
