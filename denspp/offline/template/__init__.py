"""Automatically exports all pipeline_v* modules

Note: name collisions between submodules are possible and will overwrite
previously imported names
"""

import os
import pkgutil
import re
from importlib import import_module

__all__ = []

package_paths = globals().get("__path__", [os.path.dirname(__file__)])

# only modules named "pipeline_v" + a number
module_pattern = re.compile(r"^pipeline_v\d+$")

for _finder, name, ispkg in pkgutil.iter_modules(package_paths):
    if not module_pattern.match(name):
        continue

    module = import_module(f".{name}", package=__package__)

    expected_class = "".join(part.capitalize() for part in name.split("_"))
    if hasattr(module, expected_class):
        cls = getattr(module, expected_class)
        globals()[expected_class] = cls
        __all__.append(expected_class)

__all__ = sorted(set(__all__))
