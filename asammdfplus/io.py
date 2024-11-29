import logging
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Literal,
    Sequence,
)

import pandas as pd

from ._original import MDF, Signal
from ._typing import SIG_SIGNS, PathOrPaths
from ._utils import (
    concat_dataframes_with_sequence,
    df_factory,
    get_filepaths,
)

logger = logging.getLogger(__name__)


def to_dataframe(
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

    Returns:
        The path to the saved DataFrame.
    """
    if include and isinstance(include, str):
        include = (include,)
    elif exclude and isinstance(exclude, str):
        exclude = (exclude,)

    if isinstance(fpath_or_mdf, MDF):
        # If an MDF object is given, extract the file path
        fpath = Path(fpath_or_mdf.name)
    else:
        # If a file path is given, convert it to a Path object
        fpath = Path(fpath_or_mdf)

    # Set the output file path
    out_fpath: Path = fpath.with_suffix(compression_suffix)

    if not ignore_existing and out_fpath.exists():
        # Load previously saved DataFrame
        logger.info(f"- {fpath_or_mdf} already exists, skipping")
        return pd.read_pickle(out_fpath)

    # Create an MDF object
    if isinstance(fpath_or_mdf, MDF):
        mdf: MDF = fpath_or_mdf
    else:
        mdf = MDF(fpath, raise_on_multiple_occurrences=False)

    df: pd.DataFrame = df_factory(
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
    del mdf
    if write_to_disk:
        df.to_pickle(out_fpath)
        logger.info(
            f"- Saved to {out_fpath} ({df.shape[0]} rows, {df.shape[1]} columns)"
        )
    return df


def combine_as_dataframe(
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
    **kwargs: Any,  # For compatibility
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
            to_dataframe(
                fpath_or_mdf=file_path,
                raster=raster,
                compression_suffix=compression_suffix,
                reduce_memory_usage=reduce_memory_usage,
                ignore_existing=ignore_existing,
                include=include,
                exclude=exclude,
                write_to_disk=write_to_disk,
                allow_asammdf_objects=allow_asammdf_objects,
            )
            for file_path in get_filepaths(
                path_or_paths=path_or_paths, ext_or_exts=ext_or_exts
            )
        ],
        axis=axis,
        sorted_columns=sorted_columns,
    )


def from_dataframe(fpath_or_df: str | Path | pd.DataFrame) -> MDF:
    """Load DataFrame, ensure unique column names, and save as MDF.

    Args:
        fpath_or_df: Path to the DataFrame file or a DataFrame object.

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
