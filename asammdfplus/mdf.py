import logging
import os
import re
from functools import cached_property, reduce
from pathlib import Path
from typing import (
    Iterable,
    Iterator,
    Literal,
    Optional,
    Self,
    Sequence,
    cast,
    overload,
    override,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._original import (
    MDF,
    MDF3,
    MDF4,
    BusType,
    ChannelsDB,
    ChannelsType,
    DbcFileType,
    EmptyChannelsType,
    InputType,
    RasterType,
    Signal,
    master_using_raster,
)
from ._typing import MDF_SUFFIXES
from ._utils import (
    get_channel_names_with_device,
    get_channel_names_without_device,
    hijack_channels,
    proxy_function_as_method,
)
from .io import combine_as_dataframe, from_dataframe
from .mda import CSTPlotConfig, plot, plot_cst

logger = logging.getLogger(__name__)


class MDFPlus(MDF):
    """A subclass of MDF that provides additional functionalities."""

    _mdf: MDF3 | MDF4
    name: Path
    channels_db: ChannelsDB

    ################
    # Magicmethods #
    ################
    def __getitem__(self, key: str) -> pd.Series:
        matches: set[str] = self.match(pattern=key, case_insensitive=True, include_device=False)
        if not matches:
            raise KeyError(f"Channel {key} not found in MDF.")
        if len(matches) > 1:
            raise KeyError(f"Multiple channels found for {key}: {', '.join(matches)}")
        return self.signal_to_series(self.get(matches.pop()))

    def __setitem__(
        self,
        key: str,
        value: int | float | dict[float, float] | Signal | pd.Series,
    ) -> None:
        if isinstance(value, pd.Series):
            signal = self.series_to_signal(key, value)
        elif isinstance(value, Signal):
            signal = value
        elif isinstance(value, (int, float)):
            signal = Signal(name=key, samples=[value], timestamps=[0.0])
        else:
            signal = Signal(
                name=key,
                samples=list(value.values()),
                timestamps=list(value.keys()),
            )
        self.__cache__[key] = signal
        # if key not in self.channels_db:
        #     self._mdf.append(signal)

    def __delitem__(self, key: str) -> None:
        if key in self.__cache__:
            del self.__cache__[key]
        else:
            raise KeyError(f"Cannot delete non-cached key: {key}")

    @override
    def __contains__(self, key: str) -> bool:
        return key in self.channel_names

    @override
    def __enter__(self) -> "MDFPlus":
        return self.inherit_from_mdf(super().__enter__())

    ####################
    # Proxy properties #
    ####################

    @property
    @override
    @proxy_function_as_method
    def plot(self):
        return plot

    @property
    def combine_as_dataframe(self):
        return combine_as_dataframe

    @property
    def from_dataframe(self):
        return from_dataframe

    ###########
    # Methods #
    ###########

    def match(
        self,
        pattern: str,
        case_insensitive: bool = True,
        include_device: bool | None = True,
        verbose: bool = False,
    ) -> set[str]:
        """Match the channel names with the pattern.

        Args:
            pattern: The pattern to match.
            case_insensitive: Whether to ignore the case of the pattern.
            include_device: Whether to include the device name in the channel name.
            verbose: Whether to print the matched channel names.

        Returns:
            The matched channel names."""
        compiled_pattern: re.Pattern[str] = re.compile(  # wildcard pattern
            "^" + re.escape(pattern).replace("\\*", ".*") + "$",
            flags=re.IGNORECASE if case_insensitive else 0,
        )
        matches: list[str] = [name for name in self.channel_names if compiled_pattern.search(name)]
        names: set[str]
        if include_device is None:
            names = set(matches)
        elif include_device:
            names = get_channel_names_with_device(matches)
        else:
            names = get_channel_names_without_device(matches)
        if verbose:
            for channel_name in names:
                if "\\" in channel_name:
                    name, device = channel_name.split("\\", 1)
                    print(f"- \033[36m{name}\033[34m\\{device}\033[0m")
                else:
                    print(f"- \033[36m{channel_name}\033[0m")
        return names

    @override
    def to_dataframe(
        self,
        channels: ChannelsType | None = None,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        keep_arrays: bool = False,
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        use_interpolation: bool = True,
        only_basenames: bool = False,
        interpolate_outwards_with_nan: bool = False,
        numeric_1D_only: bool = False,
        progress=None,
        use_polars: bool = False,
    ) -> pd.DataFrame:
        df: pd.DataFrame
        cached_signals: dict[str, Signal]
        non_cached_channels: Optional[ChannelsType]
        if channels is not None:
            # If there are channels to be constrained, first hijack the cached signals
            # and get the non-cached channels
            cached_signals, non_cached_channels = hijack_channels(self, channels)
        else:
            # If there are no channels to be constrained, just use the all signals
            cached_signals = self.__cache__
            non_cached_channels = None
        if non_cached_channels is None or non_cached_channels:
            # If there are no constraints or there are non-cached channels,
            # just use the original method to get the DataFrame.
            df = super().to_dataframe(
                channels=non_cached_channels,
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
                numeric_1D_only=numeric_1D_only,
                progress=progress,
                use_polars=use_polars,
            )
            # Interpolate the cached signals to the timestamps of the non-cached channels
            if cached_signals:
                master: np.ndarray = df.index.to_numpy()
                for name, signal in cached_signals.items():
                    df[name] = np.asarray(signal.interp(master).samples)
        else:
            # If there are constraints and all channels are cached,
            # just interpolate the cached signals to the timestamps of the master signal.
            if raster is not None:
                try:
                    raster = float(raster)
                    assert raster > 0
                except (TypeError, ValueError):
                    if isinstance(raster, str):
                        raster = self.get(raster).timestamps
                    else:
                        raster = np.array(raster)
                else:
                    raster = self.master_using_raster(raster)
                master = raster
            else:
                masters = {index: self.get_master(index) for index in self.virtual_groups}
                if masters:
                    master = reduce(np.union1d, masters.values())
                else:
                    master = np.array([], dtype="<f4")

                del masters

            idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
            master = master[idx]
            df = pd.DataFrame(index=master)
            for name, signal in cached_signals.items():
                df[name] = np.asarray(signal.interp(master).samples)

        return df

    @override
    def cut(
        self,
        start: float | None = None,
        stop: float | None = None,
        whence: int = 0,
        version: str | None = None,
        include_ends: bool = True,
        time_from_zero: bool = False,
        progress=None,
        *,
        timestamps: list[tuple[float, float]] | None = None,
        sync: bool = True,
        direct_timestamp_continuation: bool = True,
    ) -> "MDFPlus":
        """If the timestamps is None, cut the MDF by the (start, stop, whence) arguments.
        Otherwise, cut the MDF by the `timestamps: list[tuple[float, float]]`, and concatenate them as a new MDF.
        """
        if timestamps is None:
            # Simply cut the all MDF sources (start, stop, whence)
            new = self.inherit_from_mdf(
                super().cut(
                    start=start,
                    stop=stop,
                    whence=whence,
                    version=version,
                    include_ends=include_ends,
                    time_from_zero=time_from_zero,
                    progress=progress,
                )
            )
            new.name = self.name
            return new

        if not timestamps:  # Length check for the timestamps
            raise ValueError("No timestamps to cut")

        # Cut the MDF by the list of timestamps, and concatenate them as a new MDF
        kwargs: dict = {"version": version} if version else {}
        return self.concatenate(
            [
                super().cut(
                    start=start,
                    stop=stop,
                    whence=whence,
                    version=version,
                    include_ends=include_ends,
                    time_from_zero=time_from_zero,
                    progress=progress,
                )
                for start, stop in timestamps
            ],
            sync=sync,
            direct_timestamp_continuation=direct_timestamp_continuation,
            progress=progress,
            **kwargs,
        )

    def get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: RasterType | None = None,
        data: bytes | tuple[bytes, int, int | None] | None = None,
        raw: bool = False,
        ignore_invalidation_bits: bool = False,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_channel_validation: bool = False,
    ) -> Signal:
        """Get signal by the name or the group and the index.
        The only difference from the original method is that it caches the signal by the name.
        """
        if name in self.__cache__:
            return self.__cache__[name]
        else:
            if isinstance(self._mdf, MDF3):
                data = cast(tuple[bytes, int, int | None] | None, data)
                signal: Signal = self._mdf.get(
                    name=name,
                    group=group,
                    index=index,
                    raster=raster,
                    data=data,
                    raw=raw,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    record_offset=record_offset,
                    record_count=record_count,
                    skip_channel_validation=skip_channel_validation,
                )
            elif isinstance(self._mdf, MDF4):
                data = cast(bytes | None, data)
                signal: Signal = self._mdf.get(
                    name=name,
                    group=group,
                    index=index,
                    raster=raster,
                    data=data,
                    raw=raw,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    record_offset=record_offset,
                    record_count=record_count,
                    skip_channel_validation=skip_channel_validation,
                )
            else:
                raise TypeError(f"Unsupported MDF version: {type(self._mdf)}")

            if name:
                self.__cache__[name] = signal
            return signal

    def master_using_raster(self, raster: RasterType) -> np.ndarray:
        return master_using_raster(mdf=self._mdf, raster=raster)

    @override
    def iter_channels(
        self,
        skip_master: bool = True,
        copy_master: bool = True,
        raw: bool | dict[str, bool] = False,
    ) -> Iterator[Signal]:
        for signal in super().iter_channels(skip_master=skip_master, copy_master=copy_master, raw=raw):
            yield self.__cache__.get(signal.name, signal)

    @override
    def select(
        self,
        channels: ChannelsType,
        record_offset: int = 0,
        raw: bool | dict[str, bool] = False,
        copy_master: bool = True,
        ignore_value2text_conversions: bool = False,
        record_count: int | None = None,
        validate: bool = False,
    ) -> list[Signal]:
        hijacked_signals, non_hijacked_channels = hijack_channels(self, channels)
        return super().select(
            channels=non_hijacked_channels,
            record_offset=record_offset,
            raw=raw,
            copy_master=copy_master,
            ignore_value2text_conversions=ignore_value2text_conversions,
            record_count=record_count,
            validate=validate,
        ) + list(hijacked_signals.values())

    def inherit_from_mdf(self: "MDFPlus | None", mdf: MDF) -> "MDFPlus":
        new = MDFPlus()
        new.__dict__.update(mdf.__dict__)
        if self is not None:
            new.__cache__.update(self.__cache__)
        return new

    @override
    def filter(
        self,
        channels: ChannelsType,
        version: str | None = None,
        progress=None,
    ) -> "MDFPlus":
        new = self.inherit_from_mdf(
            super().filter(
                channels=channels,
                version=version,
                progress=progress,
            )
        )
        new.name = self.name
        new.__cache__.clear()
        for name, signal in self.__cache__.items():
            if name in channels:
                new.__cache__[name] = signal
        return new

    def get_constant_signal(
        self,
        name: str,
        value: float,
        startend: Optional[tuple[float, float]] = None,
        unit: str = "",
        comment: str = "",
    ) -> Signal:
        """Make a constant signal with the given start, end, and value."""
        if startend is None:
            start, end = self.startend
        else:
            start, end = startend
        return Signal(
            samples=np.array([value, value]),
            timestamps=np.array([start, end]),
            name=name,
            unit=unit,
            comment=comment,
        )

    def try_get_signal(self, signal_name: str, notify_if_not_exists: bool = False) -> Signal | None:
        """Try to get a signal from the MDF file. If the signal does not exist, return None."""
        try:
            return self.get(signal_name)
        except Exception:
            if notify_if_not_exists:
                logger.error(f"Signal {signal_name} does not exist in the MDF file.")
            return None

    ##############
    # Properties #
    ##############

    @property
    def channel_names(self) -> set[str]:
        channel_names = set(self.channels_db.keys())
        channel_names.update(self.__cache__.keys())
        return channel_names

    @property
    def channel_names_with_device(self) -> set[str]:
        return get_channel_names_with_device(self.channels_db.keys(), self.__cache__.keys())

    @property
    def channel_names_without_device(self) -> set[str]:
        return get_channel_names_without_device(self.channels_db.keys(), self.__cache__.keys())

    @property
    def startend(self) -> tuple[float, float]:
        """Get the start and end timestamps of the MDF file."""
        sig = next(sig for sig in self.iter_channels() if sig.timestamps.size > 0)
        return float(sig.timestamps[0]), float(sig.timestamps[-1])

    @cached_property
    def __cache__(self) -> dict[str, Signal]:
        return {}

    ################
    # Classmethods #
    ################

    @classmethod
    def try_from_file(cls, path: str | Path) -> Self | None:
        try:
            instance = cls(path)
            return instance
        except Exception:
            return None

    @classmethod
    def from_directory(cls, directory: str | Path) -> list[Self]:
        result: list[Self] = []
        for dirpath, _, filenames in os.walk(directory):
            for file in filenames:
                if Path(file).suffix.lower() not in MDF_SUFFIXES:
                    continue
                instance = cls.try_from_file(Path(dirpath) / file)
                if instance is None:
                    continue
                result.append(instance)
        return result

    #################
    # Staticmethods #
    #################

    @staticmethod
    def plot_cst(
        cst_plot_config: CSTPlotConfig,
    ) -> tuple[Figure, list[Axes], dict[str, "MDFPlus"]]:
        figs, axes, mdf_dict = plot_cst(cst_plot_config=cst_plot_config)
        mdfplus_dict: dict[str, MDFPlus] = {k: MDFPlus.inherit_from_mdf(self=None, mdf=v) for k, v in mdf_dict.items()}
        return figs, axes, mdfplus_dict

    @staticmethod
    def signal_to_series(signal: Signal) -> pd.Series:
        return pd.Series(
            data=np.asarray(signal.samples),
            index=signal.timestamps,
            name=signal.name,
        )

    @staticmethod
    def series_to_signal(name: str, series: pd.Series) -> Signal:
        bit_count = 1 if series.dtype == "bool" else None
        return Signal(
            name=name,
            timestamps=np.asarray(series.index.values),
            samples=np.asarray(series.values),
            bit_count=bit_count,
        )

    @staticmethod
    def cut_signal(signal: Signal, intervals: list[tuple[float, float]]) -> list[Signal]:
        """Cut the signal into intervals.
        The intervals are a list of tuples, where each tuple is a start and end point.
        The start and end points are in seconds.

        Returns a list of signals, each signal is a cut of the original signal.
        """
        return [signal.cut(start, end) for start, end in intervals]

    @staticmethod
    @overload
    def get_timestamps(
        mask: "pd.Series[bool]",
        as_int: Literal[False] = ...,
        inplace_mask: bool = ...,
    ) -> list[tuple[float, float]]: ...

    @staticmethod
    @overload
    def get_timestamps(
        mask: "pd.Series[bool]",
        as_int: Literal[True],
        inplace_mask: bool = ...,
    ) -> list[tuple[int, int]]: ...

    @staticmethod
    def get_timestamps(
        mask: "pd.Series[bool]",
        as_int: bool = False,
        inplace_mask: bool = True,
    ) -> list[tuple[float, float]] | list[tuple[int, int]]:
        """Get the timestamps or positional indices of the mask from the condition series.

        Args:
            mask (pd.Series[bool]): The condition series.
            inplace_mask (bool, optional): Whether to modify the mask in place. Defaults to True.
            as_int (bool, optional):
                - If True, return positional integer indices.
                - If False, return float timestamps based on the index labels.
                Defaults to False.

        Returns:
            Union[List[Tuple[int, int]], List[Tuple[float, float]]]:
                - List of start and end tuples as integers if `as_int=True`.
                - List of start and end tuples as floats if `as_int=False`.
        """
        if inplace_mask:
            mask.ffill(inplace=True)
            mask.fillna(False, inplace=True)
        else:
            mask = mask.ffill().fillna(False)

        mask = mask.astype(bool)
        if as_int:
            # 위치 기반 인덱스 사용
            positions = range(len(mask))
            mask_pos = pd.Series(mask.values, index=positions, dtype=bool)
            edge_fall = mask_pos[(~mask_pos) & mask_pos.shift(1, fill_value=mask_pos.iloc[0])].index
            edge_rise = mask_pos[mask_pos & ~mask_pos.shift(1, fill_value=mask_pos.iloc[0])].index
            if mask_pos.iloc[0]:
                edge_rise = pd.Index([0]).append(edge_rise)
            if mask_pos.iloc[-1]:
                edge_fall = edge_fall.append(pd.Index([len(mask) - 1]))
            if not edge_rise.empty and not edge_fall.empty:
                return [(s, e) for s, e in zip(edge_rise, edge_fall) if s < e]
            return []
        else:
            # 타임스탬프 기반 인덱스 사용
            edge_fall = mask[~mask & mask.shift(1, fill_value=mask.iloc[0])].index.astype(float)
            edge_rise = mask[mask & ~mask.shift(1, fill_value=mask.iloc[0])].index.astype(float)
            mask_idx: npt.NDArray[np.float64] = mask.index.to_numpy(dtype=np.float64).ravel()
            if mask.iloc[0]:
                edge_rise = edge_rise.insert(0, float(mask_idx[0]))
            if mask.iloc[-1]:
                edge_fall = edge_fall.append(pd.Index([float(mask_idx[-1])]))
            if not edge_rise.empty and not edge_fall.empty:
                return [(s, e) for s, e in zip(edge_rise, edge_fall) if s < e]
            return []

    @staticmethod
    def get_lab_measurements(lab_content: str) -> set[str]:
        """
        LAB 파일 내용에서 RAMCELL 섹션의 변수명 리스트를 추출하는 함수

        Args:
            lab_content (str): LAB 파일의 전체 내용

        Returns:
            set[str]: RAMCELL 변수명 리스트

        Raises:
            ValueError: 입력이 문자열이 아니거나 RAMCELL 섹션이 없는 경우
        """
        # 입력값 타입 검증
        if not isinstance(lab_content, str):
            raise ValueError(f"LAB 파일 내용이 문자열이 아닙니다: {lab_content}")
        ramcell_section: set[str] = set()
        reading_ramcell: bool = False

        # 줄 단위로 처리
        for line in lab_content.splitlines():
            line: str = line.strip()  # 앞뒤 공백 제거

            # RAMCELL 섹션 시작 확인
            if line == "[RAMCELL]":
                reading_ramcell = True
                continue

            # RAMCELL 섹션 종료 확인 (빈 줄이나 새로운 섹션)
            elif reading_ramcell and (not line or line.startswith("[")):
                break

            # RAMCELL 변수 처리
            if reading_ramcell and line:
                variable_name: str = line.split(";")[0]  # 세미콜론으로 분리하여 변수명만 추출
                ramcell_section.add(variable_name)

        # RAMCELL 섹션이 없거나 비어있는 경우
        if not ramcell_section:
            raise ValueError("LAB 파일에서 RAMCELL 섹션을 찾을 수 없거나 비어있습니다")

        return ramcell_section

    ########################################
    # Overrides without funcitonal changes #
    ########################################

    @override
    @staticmethod
    def stack(
        files: Sequence[MDF | InputType],
        version: str = "4.10",
        sync: bool = True,
        progress=None,
        **kwargs,
    ) -> "MDFPlus":
        return MDFPlus.inherit_from_mdf(
            None,
            MDF.stack(
                files,
                version=version,
                sync=sync,
                progress=progress,
                **kwargs,
            ),
        )

    @override
    @staticmethod
    def concatenate(
        files: Sequence[MDF | InputType],
        version: str = "4.10",
        sync: bool = True,
        add_samples_origin: bool = False,
        direct_timestamp_continuation: bool = True,
        progress=None,
        add_comments=True,
        **kwargs,
    ) -> "MDFPlus":
        return MDFPlus.inherit_from_mdf(
            None,
            MDF.concatenate(
                files,
                version=version,
                sync=sync,
                add_samples_origin=add_samples_origin,
                direct_timestamp_continuation=direct_timestamp_continuation,
                progress=progress,
                **kwargs,
            ),
        )

    @override
    def extract_bus_logging(
        self,
        database_files: dict[BusType, Iterable[DbcFileType]],
        version: str | None = None,
        ignore_invalid_signals: bool | None = None,
        consolidated_j1939: bool | None = None,
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress=None,
    ) -> "MDFPlus":
        new = self.inherit_from_mdf(
            super().extract_bus_logging(
                database_files=database_files,
                version=version,
                ignore_invalid_signals=ignore_invalid_signals,
                consolidated_j1939=consolidated_j1939,
                ignore_value2text_conversion=ignore_value2text_conversion,
                prefix=prefix,
                progress=progress,
            )
        )
        new.name = self.name
        return new

    @override
    def cleanup_timestamps(
        self,
        minimum: float,
        maximum: float,
        exp_min: int = -15,
        exp_max: int = 15,
        version: str | None = None,
        progress=None,
    ) -> "MDFPlus":
        new = self.inherit_from_mdf(
            super().cleanup_timestamps(
                minimum=minimum,
                maximum=maximum,
                exp_min=exp_min,
                exp_max=exp_max,
                version=version,
                progress=progress,
            )
        )
        new.name = self.name
        return new

    @override
    def convert(self, version: str, progress=None) -> "MDFPlus":
        new = self.inherit_from_mdf(super().convert(version=version, progress=progress))
        new.name = self.name
        return new

    @override
    def resample(
        self,
        raster: RasterType,
        version: str | None = None,
        time_from_zero: bool = False,
        progress=None,
    ) -> "MDFPlus":
        new = self.inherit_from_mdf(
            super().resample(
                raster=raster,
                version=version,
                time_from_zero=time_from_zero,
                progress=progress,
            )
        )
        new.name = self.name
        return new
