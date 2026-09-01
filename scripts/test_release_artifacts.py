#!/usr/bin/env python3
"""Install wheel and sdist into empty environments and smoke-test public CLIs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _artifact(dist_dir: Path, pattern: str, label: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label} in {dist_dir}, found {len(matches)}")
    return matches[0].resolve()


def _smoke(uv: str, artifact: Path, requirements: Path, label: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"fugal-{label}-") as raw:
        root = Path(raw)
        environment = root / ".venv"
        _run([uv, "venv", "--python", sys.executable, str(environment)])
        python = environment / "bin" / "python"
        dependency_command = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--index",
            "https://download.pytorch.org/whl/cpu",
            "--default-index",
            "https://pypi.org/simple",
            "--index-strategy",
            "unsafe-best-match",
            "--requirement",
            str(requirements),
        ]
        _run(dependency_command)
        _run([
            uv, "pip", "install", "--python", str(python), "--no-deps", str(artifact)
        ])
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        clean_env["PATH"] = f"{environment / 'bin'}:{clean_env.get('PATH', '')}"
        commands = [
            "fugal-validator",
            "fugal-validator-v2",
            "fugal-miner",
            "fugal-train",
            "fugal-verify-epoch",
        ]
        for command in commands:
            _run([str(environment / "bin" / command), "--help"], cwd=root, env=clean_env)
        _run([
            str(python),
            "-c",
            (
                "import importlib.resources, pathlib, fugal_subnet; "
                "root=importlib.resources.files('fugal_subnet'); "
                "required=['consensus-manifest.json','benchmark-registry-v2.json',"
                "'model-registry-v2.json','human-eval-cases-v2.json']; "
                "assert all(root.joinpath(name).is_file() for name in required); "
                "assert pathlib.Path(fugal_subnet.__file__).resolve().is_relative_to("
                "pathlib.Path(__import__('sys').prefix).resolve())"
            ),
        ], cwd=root, env=clean_env)
        print(f"clean {label} installation passed: {artifact.name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--lock-project", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for clean artifact smoke tests")
    dist_dir = args.dist_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="fugal-lock-export-") as raw:
        requirements = Path(raw) / "requirements.txt"
        _run([
            uv,
            "export",
            "--project",
            str(args.lock_project.resolve()),
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements),
        ])
        _smoke(uv, _artifact(dist_dir, "*.whl", "wheel"), requirements, "wheel")
        _smoke(
            uv,
            _artifact(dist_dir, "*.tar.gz", "source distribution"),
            requirements,
            "sdist",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
