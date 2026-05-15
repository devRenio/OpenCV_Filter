"""Filter package with automatic plug-in discovery.

Importing this package walks the directory and imports every ``*_filter.py``
module so that any class decorated with :func:`core.registry.register`
becomes available without requiring manual edits to this file.

Adding a new filter is therefore a one-step process:

1. Create ``filters/<your_name>_filter.py``.
2. Define a subclass of :class:`core.base_filter.BaseFilter` decorated with
   :func:`core.registry.register`.

The new filter will appear in the UI automatically the next time the app
starts.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _auto_discover_filters() -> None:
    """Import every filter module within this package.

    Modules whose name starts with an underscore are skipped, allowing for
    private helper modules that should not be auto-loaded.
    """
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        full_name = f"{__name__}.{module_name}"
        try:
            importlib.import_module(full_name)
            logger.debug("Loaded filter module: %s", full_name)
        except Exception:  # noqa: BLE001 - log and continue on plug-in errors
            logger.exception("Failed to import filter module %s", full_name)


_auto_discover_filters()
