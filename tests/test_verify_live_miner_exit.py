"""verify_live_miner exits after printing its verdict instead of hanging in
interpreter teardown (v3 testnet check: exit 124 under `timeout`)."""
import ast
import pathlib


def _src():
    return pathlib.Path(__file__).resolve().parents[1].joinpath(
        "scripts", "verify_live_miner.py").read_text()


def test_the_entry_point_exits_through_os_exit():
    tree = ast.parse(_src())
    main_guard = next(n for n in tree.body if isinstance(n, ast.If)
                      and "__main__" in ast.unparse(n.test))
    assert "_exit_now(main())" in ast.unparse(main_guard)
    exit_now = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_exit_now")
    assert "os._exit(code)" in ast.unparse(exit_now)
