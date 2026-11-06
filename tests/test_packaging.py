"""The wheel ships every pinned consensus file the package reads at runtime.

Found 2026-11-06: pyproject's package-data listed pool_manifest.json but not
pool_manifest_v2.json, so a wheel install carried the v1 manifest and not the
v2 one. The Docker image installs the project from its source tree, so it was
unaffected; a `pip install` of the built wheel was not.
"""
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _package_data() -> dict[str, list[str]]:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["setuptools"]["package-data"]


def test_both_pool_manifests_are_package_data():
    data = _package_data()["fugal_subnet.benchmarks"]
    for name in ("pool_manifest.json", "pool_manifest_v2.json"):
        assert name in data, name
        assert (ROOT / "fugal_subnet" / "benchmarks" / name).exists(), name


def test_every_runtime_data_file_under_the_package_is_shipped():
    """Every non-Python file the package tree holds is matched by a
    package-data pattern (so a new pinned file cannot be forgotten)."""
    import fnmatch

    data = _package_data()
    pkg = ROOT / "fugal_subnet"
    unshipped = []
    for path in pkg.rglob("*"):
        if not path.is_file() or path.suffix in (".py", ".pyc") or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg)
        # vendor/source_proof.json is read by scripts/check_success_vendor.py in CI, not at runtime
        if str(rel) == "vendor/source_proof.json":
            continue
        shipped = False
        for package, patterns in data.items():
            sub = package.removeprefix("fugal_subnet").lstrip(".").replace(".", "/")
            for pattern in patterns:
                full = f"{sub}/{pattern}" if sub else pattern
                if fnmatch.fnmatch(str(rel), full):
                    shipped = True
        if not shipped:
            unshipped.append(str(rel))
    assert not unshipped, unshipped
