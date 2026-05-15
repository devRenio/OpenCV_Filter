"""Core package containing the abstract filter interface and the filter registry."""

from core.base_filter import BaseFilter
from core.registry import FilterRegistry, register

__all__ = ["BaseFilter", "FilterRegistry", "register"]
