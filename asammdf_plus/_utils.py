import logging
import re
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Concatenate,
    Iterable,
    ParamSpec,
    TypeVar,
)

from asammdf import Signal
from asammdf.types import ChannelsType

if TYPE_CHECKING:
    from .mdf import MDFPlus

SelfType = TypeVar("SelfType")
ReturnType = TypeVar("ReturnType")
P = ParamSpec("P")
V = TypeVar("V")

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
