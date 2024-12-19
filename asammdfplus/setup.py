import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeAlias

import matplotlib.pyplot as plt
from matplotlib import font_manager

module_logger = logging.getLogger(__name__)

PathLike: TypeAlias = os.PathLike | str


class CustomFormatter(logging.Formatter):
    """
    Custom formatter to produce the desired log format.
    """

    def __init__(
        self,
        time_fmt: str = "%y-%m-%d %H:%M:%S",
        include_stack_info: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_fmt: str = time_fmt
        self.include_stack_info: bool = include_stack_info

    def format(self, record):
        if record.exc_info and self.include_stack_info:
            # Include stack trace in log
            return f"[{datetime.now().strftime(self.time_fmt)}][{record.levelname}][{record.name}] {record.getMessage()}\n{self.formatException(record.exc_info)}"  # noqa: E501
        else:
            # Do not include stack trace
            return f"[{datetime.now().strftime(self.time_fmt)}][{record.levelname}][{record.name}] {record.getMessage()}"  # noqa: E501


def setup_logger(
    logger_name: str | None = None,
    logger_level: int = logging.DEBUG,
    use_console: bool = True,
    use_file: bool = True,
    stream_handler_level: int = logging.INFO,
    stream_include_stack_info: bool = False,
    stream_time_fmt: str = "%y-%m-%d %H:%M:%S",
    file_handler_level: int = logging.DEBUG,
    file_include_stack_info: bool = True,
    file_time_fmt: str = "%y-%m-%d %H:%M:%S",
    file_name_fmt: str = "./logs/%y%m%d.log",
) -> logging.Logger:
    """
    Configures the logger with the given settings and returns it.
    """
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        logger.setLevel(logger_level)

        if use_console:
            # Stream Handler (for console output)
            stream_formatter = CustomFormatter(
                include_stack_info=stream_include_stack_info,
                time_fmt=stream_time_fmt,
            )
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(stream_formatter)
            stream_handler.setLevel(stream_handler_level)
            logger.addHandler(stream_handler)

        if use_file and file_name_fmt:
            # File Handler (for file logging)
            file_formatter = CustomFormatter(
                include_stack_info=file_include_stack_info,
                time_fmt=file_time_fmt,
            )
            filepath = Path(datetime.now().strftime(file_name_fmt))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(file_handler_level)
            logger.addHandler(file_handler)
    else:
        logger.warning("Logger already has handlers. Skipping setup.")

    return logger


def setup_jupyter_notebook(
    font_path: Optional[PathLike] = None,
    logger_name: str | None = None,
    logger_level: int = logging.DEBUG,
    use_console: bool = True,
    use_file: bool = True,
    stream_handler_level: int = logging.INFO,
    stream_include_stack_info: bool = False,
    stream_time_fmt: str = "%y-%m-%d %H:%M:%S",
    file_handler_level: int = logging.DEBUG,
    file_include_stack_info: bool = True,
    file_time_fmt: str = "%y-%m-%d %H:%M:%S",
    file_name_fmt: str = "./logs/%y%m%d.log",
) -> None:
    """Setup notebook environment.

    1. Set unicode minus to False (to use minus sign instead of hyphen).
    2. Setup logger.
    3. Add font file to font manager."""
    plt.rcParams["axes.unicode_minus"] = False

    try:
        logger = setup_logger(
            logger_name=logger_name,
            logger_level=logger_level,
            use_console=use_console,
            use_file=use_file,
            stream_handler_level=stream_handler_level,
            stream_include_stack_info=stream_include_stack_info,
            stream_time_fmt=stream_time_fmt,
            file_handler_level=file_handler_level,
            file_include_stack_info=file_include_stack_info,
            file_time_fmt=file_time_fmt,
            file_name_fmt=file_name_fmt,
        )
        logger.info(f"Logger setup completed: {logger}")
    except Exception as e:
        module_logger.error(f"Failed to setup logger: {e}")

    if font_path is not None and Path(font_path).is_file():
        try:
            for font_file in font_manager.findSystemFonts(
                fontpaths=[Path(font_path).parent]
            ):
                font_manager.fontManager.addfont(font_file)

            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=font_path
            ).get_name()
            module_logger.info(f"Font file added: {font_path}")
        except Exception as e:
            module_logger.error(f"Failed to add font file: {e}")
