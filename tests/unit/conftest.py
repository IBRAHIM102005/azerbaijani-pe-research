import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_sys_path_and_modules():
    """Tests in this directory monkeypatch `sys.modules["src.models"]`
    (see tests/fake_m2.py) to install a fake Yasin model package for testing
    this suite's wrapper scripts in isolation. A couple of tests also
    delete `sys.modules["src"]` outright to simulate a repo where no
    Ibrahim/Yasin/Fidan/Nihat's code exists at all.

    Snapshot and restore the actual module objects (not just whether a
    name happened to exist) around every test, so a fake installed by one
    test can never leak into another, and the repo's real `src.models`
    (and `src` itself) is always back in place afterward -- regardless of
    whether it already existed before this test ran, which it does in
    this repo, unlike in an isolated standalone checkout.
    """
    original_path = list(sys.path)
    original_src = sys.modules.get("src")
    original_src_models = sys.modules.get("src.models")
    original_models_attr = getattr(original_src, "models", None) if original_src else None

    yield

    sys.path[:] = original_path

    if original_src is not None:
        sys.modules["src"] = original_src
    else:
        sys.modules.pop("src", None)

    if original_src_models is not None:
        sys.modules["src.models"] = original_src_models
    else:
        sys.modules.pop("src.models", None)

    if original_src is not None:
        if original_models_attr is not None:
            original_src.models = original_models_attr
        elif hasattr(original_src, "models"):
            del original_src.models

    # Make sure the real top-level package backs every other test.
    if "src" not in sys.modules:
        import importlib

        importlib.import_module("src")
