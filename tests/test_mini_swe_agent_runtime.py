import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class Submitted(Exception):
    pass


class FakeInteractiveAgent:
    def __init__(self, environment, model) -> None:
        self.env = environment
        self.model = model
        self.messages = []

    def _ask_confirmation_or_interrupt(self, commands) -> None:
        pass

    def _check_for_new_task_or_submit(self, exc) -> None:
        raise exc

    def get_template_vars(self) -> dict:
        return {}

    def add_messages(self, *messages):
        self.messages.extend(messages)
        return list(messages)


class FakeEnvironment:
    def execute(self, action):
        if action["command"] == "submit":
            raise Submitted()
        return {"output": action["command"], "returncode": 0, "exception_info": ""}


class FakeModel:
    def format_observation_messages(self, message, outputs, template_vars):
        results = list(outputs)
        results.extend(
            {
                "output": "",
                "returncode": -1,
                "exception_info": "action was not executed",
            }
            for _ in range(len(message["extra"]["actions"]) - len(outputs))
        )
        return [{"extra": result} for result in results]


def _load_runtime(monkeypatch):
    interactive = ModuleType("minisweagent.agents.interactive")
    interactive.InteractiveAgent = FakeInteractiveAgent
    exceptions = ModuleType("minisweagent.exceptions")
    exceptions.Submitted = Submitted
    monkeypatch.setitem(sys.modules, "minisweagent", ModuleType("minisweagent"))
    monkeypatch.setitem(
        sys.modules, "minisweagent.agents", ModuleType("minisweagent.agents")
    )
    monkeypatch.setitem(sys.modules, "minisweagent.agents.interactive", interactive)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)

    path = (
        Path(__file__).parents[1]
        / "src/pier/agents/installed/mini_swe_agent_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("pier_minisweagent_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("commands", "returncodes"),
    [
        (["submit"], [0]),
        (["before", "submit", "after"], [0, 0, -1]),
    ],
)
def test_submit_action_gets_success_before_exit(monkeypatch, commands, returncodes):
    module = _load_runtime(monkeypatch)
    agent = module.SubmittedResultInteractiveAgent(FakeEnvironment(), FakeModel())
    message = {"extra": {"actions": [{"command": command} for command in commands]}}

    with pytest.raises(Submitted):
        agent.execute_actions(message)

    assert [item["extra"]["returncode"] for item in agent.messages] == returncodes
