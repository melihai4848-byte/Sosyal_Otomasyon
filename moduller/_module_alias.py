from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def load_numbered_module(target_filename: str):
    current_file = Path(__file__).resolve()
    target_file = current_file.with_name(target_filename)
    module_name = f"moduller._aliased_{target_file.stem}"

    module = sys.modules.get(module_name)
    if module is None:
        spec = spec_from_file_location(module_name, target_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Alias module yuklenemedi: {target_file}")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module


def reexport_module(namespace: dict, target_filename: str) -> None:
    module = load_numbered_module(target_filename)

    namespace["_aliased_module"] = module

    for name in dir(module):
        if name.startswith("__") and name not in {"__doc__", "__all__"}:
            continue
        namespace[name] = getattr(module, name)
