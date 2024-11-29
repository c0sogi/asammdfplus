from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import DTypeLike
    from pandas.core.dtypes.dtypes import ExtensionDtype

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
AstypeArg: TypeAlias = Union["ExtensionDtype", "DTypeLike"]
AstypeArgExt: TypeAlias = (
    AstypeArg
    | Literal[
        "number",
        "datetime64",
        "datetime",
        "timedelta",
        "timedelta64",
        "datetimetz",
        "datetime64[ns]",
    ]
)
AstypeArgExtList: TypeAlias = AstypeArgExt | list[AstypeArgExt]
Interval: TypeAlias = tuple[float, float]
DataFrame: TypeAlias = "pd.DataFrame"

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
LegendLocation: TypeAlias = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

ColorLike: TypeAlias = str | tuple[float, float, float, float]


class Timestamps(NamedTuple):
    start_with_offset: float  # 측정 시작 시각 [s]
    start: float  # 이벤트 시작 [s]
    in_between: float  # 이벤트 발생 [s]
    end: float  # 이벤트 끝 [s]
    end_with_offset: float  # 측정 종료 시각 [s]


class Package(NamedTuple):
    name: str
    timestamps: "np.ndarray"
    samples: "np.ndarray"
    label: str
    color: tuple[float, float, float, float]


class GroupProperty(NamedTuple):
    same_range: bool
    tickless: bool
    signals: Sequence[str]


@dataclass
class ColdStartSummaryData:
    idx: int
    car: str
    transmission: str
    temperature: float
    starting_time: float
    cranking_time: float


@dataclass
class ColdStartMetaData:
    idx: int  # 시동 인덱스 (0부터 시작)
    car: str  # 차종
    transmission: str  # 트랜스미션
    date: str  # 날짜
    fuel: str  # 연료
    ambient_temperature: str  # 외기 온도 [℃]
    cranking_time: str  # 크랭킹 시간 [s]
    starting_time: str  # 시동 시간 [s]
    average_cranking_rpm: str  # 평균 크랭킹 속도 [rpm]
    peak_rpm: str  # 최고 엔진 속도 [rpm]
    others: dict[str, str] = field(default_factory=dict)
    inferred_units: dict[str, str] = field(default_factory=dict)


@dataclass
class ColdStartCollection:
    path: Path  # 파일 경로
    metadata_list: list[ColdStartMetaData]  # 시험 전체 요약 정보
    timestamps: list["Timestamps"]  # 각 시동 마다의 시간 정보 (length=N)
    mask: list[bool]  # 시동 성공 여부 (length=N)

    @property
    def excel_data(self) -> dict[str, list[str]]:
        return {
            "파일명": [self.path.name] * len(self.metadata_list),
            "차량": [metadata.car for metadata in self.metadata_list],
            "트랜스미션": [
                metadata.transmission for metadata in self.metadata_list
            ],
            "날짜": [metadata.date for metadata in self.metadata_list],
            "연료": [metadata.fuel for metadata in self.metadata_list],
            "외기온 [℃]": [
                metadata.ambient_temperature
                for metadata in self.metadata_list
            ],
            "초폭 [s]": [
                metadata.cranking_time for metadata in self.metadata_list
            ],
            "완폭 [s]": [
                metadata.starting_time for metadata in self.metadata_list
            ],
            "최대 RPM": [
                metadata.peak_rpm for metadata in self.metadata_list
            ],
            **{
                key: [
                    metadata.others.get(key, "")
                    for metadata in self.metadata_list
                ]
                for key in self.metadata_list[0].others.keys()
            },
        }


@dataclass
class PlottingSignalYRange:
    min: float
    max: float

    def __post_init__(self) -> None:
        if not isinstance(self.min, (int, float)) or not isinstance(
            self.max, (int, float)
        ):
            raise ValueError("Invalid range")

    def to_serializable(self) -> dict[str, float]:
        return {"min": self.min, "max": self.max}


@dataclass
class ColdStartConfig:
    font_path: str = "fonts/HDHL.ttf"
    fig_size: tuple[int, int] = (10, 5)
    dpi: int = 100
    font_size: int = 18
    legend_location: Optional[str] = "upper right"
    fig_cols: int = 1
    draw_cranking_time: bool = False
    hide_ax_title: bool = True
    threshold_rpm: float = 500.0
    start_offset: float = 0.0
    end_offset: float = 10.0
    start_temp: float = -35.0
    end_temp: float = 0.0
    start_bit_debounce_steps: int = 10
    signal_mappings: dict[str, str] = field(
        default_factory=lambda: {
            # Mandatory named signals
            "Engine Speed": "Eng_N",
            "Start Bit": "Strt_c",
            # Named signals
            "Ignition": "Misf_tqI8",
            "Amb. Temp.": "AmbT_t",
            # Optional non-named signals
            "Soak Time": "Soak_ti",
            "냉각수온": "EngT_t",
            "오일온": "OilT_tSmp8",
            "모델 오일온": "OilT_tMdl8",
            "SOC": "Batt_SOC",
            "배터리 전압": "Batt_u16",
        }
    )
    reference_points: dict[float, float] = field(
        default_factory=lambda: {
            -30: 8,
            -25: 5,
            -20: 3,
            -15: 1.75,
            -7: 1.15,
            -5: 1.05,
            0: 1,
        }
    )
    plotting_signals: dict[str, list[str]] = field(
        default_factory=lambda: {
            "배터리 전압 / 연료 압력": ["Batt_u16", "Fu_p"],
            "엔진 RPM / 기준 RPM": ["Epm_N6Tooth", "IdlnT_nSetP"],
            "Lambda / Lambda control output": ["Lambda", "Lam_kCL"],
            "점화각": ["Ign_AgOut"],
            "촉매 히팅 Bit": ["CatH_cActv"],
            "시동 Bit": ["Strt_c"],
        }
    )
    ylims: dict[str, PlottingSignalYRange] = field(
        default_factory=lambda: {
            "Batt_u16": PlottingSignalYRange(min=8, max=14),
            "Fu_p": PlottingSignalYRange(min=0, max=24),
            "Epm_N6Tooth": PlottingSignalYRange(min=0, max=2000),
            "IdlnT_nSetP": PlottingSignalYRange(min=0, max=2000),
            "Lambda": PlottingSignalYRange(min=0.5, max=1.5),
            "Lam_kCL": PlottingSignalYRange(min=0.5, max=1.5),
            "Ign_AgOut": PlottingSignalYRange(min=-20, max=20),
        }
    )

    def to_serializable(self) -> dict:
        data: dict[str, Any] = asdict(self)
        data["ylims"] = {
            key: value.to_serializable() for key, value in self.ylims.items()
        }
        data["reference_points"] = {
            str(k): v for k, v in self.reference_points.items()
        }

        return data
