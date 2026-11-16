"""The rehearsal stub answers v3 calls the way the v3 harness sends them: the
shared system prompt first, the question as the last user turn, one menu entry
per (model, reasoning effort), and code questions as a fenced program the real
code grader runs. Without these a v3 rehearsal grades every answer wrong."""
import importlib.util
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stub_upstream", ROOT / "scripts" / "stub_upstream.py")
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)


@pytest.fixture()
def server():
    H = stub.Handler
    saved = (H.gold, H.bench, H.solution, H.skill)
    H.gold = {"What is 2+3? End your response with the final answer as a plain number.": "5"}
    H.bench = {k: "mmlu" for k in H.gold}
    H.solution = {"Read two integers and print their sum.": "a, b = map(int, input().split())\n"
                                                             "print(a + b)"}
    H.bench["Read two integers and print their sum."] = "code"
    H.skill = {"m/x": {"mmlu": 1.0, "code": 1.0, "": 1.0},
               "m/x@low": {"mmlu": 0.0, "code": 0.0, "": 0.0},
               "m/x@high": {"mmlu": 1.0, "code": 1.0, "": 1.0}}
    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    H.gold, H.bench, H.solution, H.skill = saved


def ask(port, prompt, model="m/x", effort=None):
    body = {"model": model, "messages": [{"role": "system", "content": "You are Fugal."},
                                         {"role": "user", "content": prompt}]}
    if effort:
        body["reasoning"] = {"effort": effort}
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urlopen(req, timeout=10).read())["choices"][0]["message"]["content"]


def test_the_question_is_the_last_user_turn_not_the_system_prompt(server):
    q = "What is 2+3? End your response with the final answer as a plain number."
    assert ask(server, q) == "5"


def test_each_effort_is_its_own_entry(server):
    q = "What is 2+3? End your response with the final answer as a plain number."
    assert ask(server, q, effort="high") == "5"
    assert ask(server, q, effort="low") == "0"


def test_a_code_question_gets_a_program_the_code_grader_passes(server):
    from fugal_subnet import graders_code

    reply = ask(server, "Read two integers and print their sum.")
    assert reply.startswith("```python")
    task = {"checker": {"id": "exec_stdio",
                        "params": {"tests": [{"input": "2 3\n", "output": "5\n"}]}}}
    if not graders_code.sandbox_selftest():
        pytest.skip("seccomp sandbox unavailable on this host")
    assert graders_code.grade(task, reply, allow_exec=True) == 1
    wrong = ask(server, "Read two integers and print their sum.", effort="low")
    assert graders_code.grade(task, wrong, allow_exec=True) == 0


def test_the_chain_rehearsal_stub_answers_v3_calls_the_same_way():
    """scripts/setup_local_testnet.StubUpstream (what dress_rehearsal.py runs)."""
    import socket

    from scripts.setup_local_testnet import StubUpstream

    q = "What is 2+3? End your response with the final answer as a plain number."
    code_q = "Read two integers and print their sum."
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    st = StubUpstream({q: "5"}, {"m/x": 1.0, "m/x@low": 0.0, "m/x@high": 1.0}, port=port,
                      solution_by_prompt={code_q: "print(sum(map(int, input().split())))"})
    st.start()
    try:
        assert ask(port, q) == "5" and ask(port, q, effort="high") == "5"
        assert ask(port, q, effort="low") == "0"
        assert ask(port, code_q).startswith("```python\nprint(sum(")
    finally:
        st.stop()
