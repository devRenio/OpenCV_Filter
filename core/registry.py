"""Singleton-based registry for image filters.

The registry acts as a central directory of all available filters. New
filters are added by decorating their class with :func:`register`, which
guarantees that simply importing the filter module is enough for it to
become available in the UI.
"""

from __future__ import annotations

from typing import Type

from core.base_filter import BaseFilter


class FilterRegistry:
    """Singleton registry that stores and dispenses filter instances.

    The class enforces the singleton pattern via :meth:`__new__` so that
    every part of the application interacts with the same registry.
    Filters are stored as a mapping from their display name to a class,
    which guarantees uniqueness while remaining easy to look up.
    """

    _instance: "FilterRegistry | None" = None
    _filters: dict[str, Type[BaseFilter]]

    def __new__(cls) -> "FilterRegistry":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._filters = {}
            cls._instance = instance
        return cls._instance

    def register(self, filter_cls: Type[BaseFilter]) -> Type[BaseFilter]:
        """Register a concrete filter class with the registry.

        Args:
            filter_cls: A subclass of :class:`BaseFilter` to be registered.

        Returns:
            The same class, unchanged. Returning the class lets this method
            be used directly as a decorator.

        Raises:
            TypeError: If ``filter_cls`` is not a subclass of
                :class:`BaseFilter`.
            ValueError: If a filter with the same display name has already
                been registered.
        """
        if not (isinstance(filter_cls, type) and issubclass(filter_cls, BaseFilter)):
            raise TypeError(
                f"{filter_cls!r} must be a subclass of BaseFilter to be registered."
            )

        display_name = getattr(filter_cls, "name", filter_cls.__name__)
        if display_name in self._filters:
            existing = self._filters[display_name]
            if existing is not filter_cls:
                raise ValueError(
                    f"A different filter named {display_name!r} is already "
                    f"registered: {existing!r}."
                )
        self._filters[display_name] = filter_cls
        return filter_cls

    def get_all_filters(self) -> list[BaseFilter]:
        """Return a fresh instance of every registered filter.

        Returns:
            A list of filter instances, sorted alphabetically by display
            name to keep the UI ordering deterministic.
        """
        return [cls() for _, cls in sorted(self._filters.items(), key=lambda kv: kv[0])]

    def get_filter_names(self) -> list[str]:
        """Return all registered filter display names in sorted order."""
        return sorted(self._filters.keys())

    def get_filter_by_name(self, name: str) -> BaseFilter:
        """Instantiate and return a registered filter looked up by name.

        Args:
            name: Display name used at registration time.

        Returns:
            A new instance of the matching filter.

        Raises:
            KeyError: If no filter is registered under ``name``.
        """
        if name not in self._filters:
            raise KeyError(f"No filter registered under the name {name!r}.")
        return self._filters[name]()

    def clear(self) -> None:
        """Remove every registered filter (useful in tests)."""
        self._filters.clear()


def register(filter_cls: Type[BaseFilter]) -> Type[BaseFilter]:
    """Decorator that registers a filter class with the global registry.

    Example:
        >>> @register
        ... class MyFilter(BaseFilter):
        ...     name = "My Filter"
        ...     description = "Does something cool."
        ...     def apply(self, image, **kwargs):
        ...         return image

    Args:
        filter_cls: The filter class to register.

    Returns:
        The same class, unchanged.
    """
    return FilterRegistry().register(filter_cls)
