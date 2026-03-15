from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any

from loguru import logger as _logger

_LOG_LEVEL: str | None = None
_LOGGER_CONFIGURED = False
_VALID_LEVELS = {
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_log_level(level: str | None) -> str:
    global _LOG_LEVEL

    if _LOG_LEVEL is None:
        candidate = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
        _LOG_LEVEL = candidate if candidate in _VALID_LEVELS else "INFO"

    return _LOG_LEVEL


def _build_suffix_parts(
    suffix: str,
    *,
    strip_file: bool,
    use_pid: bool,
) -> tuple[str, ...]:
    parts: list[str] = []

    if suffix:
        parts.append(os.path.basename(suffix) if strip_file else suffix)

    if use_pid:
        parts.insert(0, f"pid={os.getpid()}")

    return tuple(parts)


def _try_get_tp_rank() -> int | None:
    tp_info = import_module("minisgl.distributed").try_get_tp_info()
    if tp_info is None:
        return None
    return tp_info.rank


def _format_record(record: dict[str, Any]) -> str:
    extra = record["extra"]
    suffix_parts = list(extra.get("suffix_parts", ()))

    if extra.get("use_tp_rank", True):
        tp_rank = _try_get_tp_rank()
        if tp_rank is not None:
            suffix_parts.extend(("core", f"rank={tp_rank}"))

    suffix = "".join(f"|{part}" for part in suffix_parts)
    return (
        f"<bold>[{record['time']:%Y-%m-%d|%H:%M:%S}{suffix}]</bold> "
        "<level>{level: <8}</level> {message}\n"
    )


def _configure_logger(level: str) -> None:
    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED:
        return

    _logger.remove()
    _logger.add(
        sys.stdout,
        level=level,
        format=_format_record,
        colorize=True,
        backtrace=False,
        diagnose=False,
        catch=False,
    )
    _LOGGER_CONFIGURED = True


class TurboLogger:
    """Thin wrapper around loguru with rank0 helpers."""

    def __init__(self, bound_logger: Any):
        self._logger = bound_logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

    def _log_rank0(self, method: str, message: str, *args: Any, **kwargs: Any) -> None:
        tp_info = import_module("turbo.distributed").get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(self._logger, method)(message, *args, **kwargs)

    def debug_rank0(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_rank0("debug", message, *args, **kwargs)

    def info_rank0(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_rank0("info", message, *args, **kwargs)

    def warning_rank0(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_rank0("warning", message, *args, **kwargs)

    def critical_rank0(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_rank0("critical", message, *args, **kwargs)


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
) -> TurboLogger:
    """Create a module-scoped loguru logger with project formatting."""
    resolved_level = _resolve_log_level(level)
    _configure_logger(resolved_level)

    if use_pid is None:
        use_pid = _env_flag("LOG_PID")

    bound_logger = _logger.bind(
        logger_name=name,
        suffix_parts=_build_suffix_parts(
            suffix,
            strip_file=strip_file,
            use_pid=use_pid,
        ),
        use_tp_rank=use_tp_rank is not False,
    )
    return TurboLogger(bound_logger)
