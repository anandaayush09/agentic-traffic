"""Windows workaround for a real Stable-Baselines3 / PyTorch incompatibility, not a bug
in this project's own code.

`stable_baselines3.common.save_util.load_from_zip_file` reads each inner `.pth` member via
`zipfile.ZipFile.open(name, mode="r")`, a streaming, only-partially-seekable file object,
and hands that directly to `torch.load()`. On this platform (verified against
torch==2.8.0 and torch==2.14.0 -- not a version-specific regression), PyTorch's C++ zip
reader cannot reliably read from that stream and fails with a misleading "corrupted
checkpoint" error, even though the underlying weights are completely intact (verified: the
exact same bytes load without error once read fully into memory or extracted to a real
file first).

`patch_sb3_zip_loading()` monkeypatches SB3's loader with a byte-for-byte reimplementation
that changes exactly one thing: each `.pth` member is read fully into memory
(`io.BytesIO(archive.read(name))`) instead of opened as a stream, which is proven to
avoid the failure. Everything else is unchanged from SB3's own `load_from_zip_file`. Both
`stable_baselines3.common.save_util.load_from_zip_file` and
`stable_baselines3.common.base_class.load_from_zip_file` must be patched -- `base_class.py`
imports the function by name (`from ... import load_from_zip_file`), so patching only the
`save_util` module's attribute would not affect the reference `base_class.load()` actually
calls.
"""

from __future__ import annotations

import io
import os
import pathlib
import warnings
import zipfile
from typing import Any

import torch as th


def _load_from_zip_file_fixed(
    load_path: str | pathlib.Path | io.BufferedIOBase,
    load_data: bool = True,
    custom_objects: dict[str, Any] | None = None,
    device: th.device | str = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    # Deliberately reaching into SB3's internal (non-exported) helpers to faithfully
    # mirror its own loading logic -- see module docstring.
    from stable_baselines3.common.save_util import (  # type: ignore[attr-defined]
        get_device,
        json_to_data,
        open_path,
    )

    file = open_path(load_path, "r", verbose=verbose, suffix="zip")
    device = get_device(device=device)

    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            data = None
            pytorch_variables = None
            params: dict[str, Any] = {}

            if print_system_info:
                if "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())
                else:
                    warnings.warn(
                        "The model was saved with SB3 <= 1.2.0 and thus cannot print "
                        "system information.",
                        UserWarning,
                        stacklevel=2,
                    )

            if "data" in namelist and load_data:
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            pth_files = [n for n in namelist if os.path.splitext(n)[1] == ".pth"]
            for file_path in pth_files:
                # The fix: read the member fully into memory instead of
                # archive.open(file_path, mode="r")'s streaming, only-partially-
                # seekable file object -- see module docstring.
                buffer = io.BytesIO(archive.read(file_path))
                th_object = th.load(buffer, map_location=device, weights_only=True)
                if file_path in ("pytorch_variables.pth", "tensors.pth"):
                    pytorch_variables = th_object
                else:
                    params[os.path.splitext(file_path)[0]] = th_object
    except zipfile.BadZipFile as e:
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
    finally:
        if isinstance(load_path, (str, pathlib.Path)):
            file.close()

    return data, params, pytorch_variables


_PATCHED = False


def patch_sb3_zip_loading() -> None:
    """Applies the workaround. Safe to call multiple times; only the first call takes
    effect. Call this before any `stable_baselines3.<Algo>.load(...)` call.
    """
    global _PATCHED
    if _PATCHED:
        return

    import stable_baselines3.common.base_class as base_class_module
    import stable_baselines3.common.save_util as save_util_module

    # Monkeypatching internal (non-exported) attributes on purpose -- see module docstring
    # for why both modules must be patched.
    save_util_module.load_from_zip_file = _load_from_zip_file_fixed
    base_class_module.load_from_zip_file = _load_from_zip_file_fixed  # type: ignore[attr-defined]
    _PATCHED = True
