"""Reconcile the two utils3d APIs that SAM 3D Objects and MoGe v2 each need.

utils3d renamed its torch submodule from `utils3d.torch` to `utils3d.pt` and,
in the same rewrite, moved `depth_edge` and friends out of `utils3d.numpy`.
That leaves no single release that satisfies both consumers:

  * SAM 3D Objects' visualisation module imports `utils3d.numpy.depth_edge`,
    which only exists up to commit d790d33;
  * MoGe v2 calls `utils3d.pt.intrinsics_from_focal_center`, which only exists
    after the rename.

Pinning the older commit and aliasing `pt` onto `torch` satisfies both, and is
a great deal smaller than reimplementing five numpy helpers. Import this
before anything that touches either library.
"""
from __future__ import annotations

import importlib
import sys


def install() -> str:
    """Idempotent; returns a short description of what was patched."""
    import utils3d  # noqa: F401  (import for its side effects)

    try:
        importlib.import_module("utils3d.pt")
        return "utils3d.pt already present; no aliasing needed"
    except ModuleNotFoundError:
        pass

    try:
        torch_mod = importlib.import_module("utils3d.torch")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "utils3d exposes neither `pt` nor `torch`; install "
            "git+https://github.com/EasternJournalist/utils3d.git@d790d33"
        ) from exc

    sys.modules["utils3d.pt"] = torch_mod
    setattr(utils3d, "pt", torch_mod)
    return "aliased utils3d.pt -> utils3d.torch"


DESCRIPTION = install()
