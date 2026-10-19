import logging
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from pier.cli.main import app
from pier.environments.daytona import DaytonaEnvironment
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.task import Task
from pier.models.trial.config import AgentConfig, EnvironmentConfig
from pier.models.trial.paths import TrialPaths
from pier.trial.execution import TrialExecution


@pytest.mark.parametrize("from_config", [False, True])
def test_cli_merges_agent_hosts(monkeypatch, tmp_path, from_config):
    import json
    from pier.environments.factory import EnvironmentFactory
    from pier.job import Job

    captured = []

    async def capture(config):
        captured.append(config)
        raise RuntimeError("captured job")

    monkeypatch.setattr(Job, "create", capture)
    monkeypatch.setattr(EnvironmentFactory, "run_preflight", lambda **kwargs: None)
    args = [
        "run",
        "--allow-agent-host",
        "Endpoint.TEST",
        "--allow-agent-host",
        "endpoint.test",
        "--yes",
    ]
    if from_config:
        config_path = tmp_path / "job.json"
        config_path.write_text(
            json.dumps(
                {
                    "agents": [
                        {
                            "name": "mini-swe-agent",
                            "extra_allowed_hosts": ["existing.test"],
                        }
                    ]
                }
            )
        )
        args += ["--config", str(config_path)]
    else:
        args += ["--agent", "mini-swe-agent", "--model", "fireworks_ai/model"]
    result = CliRunner().invoke(app, args)
    assert str(result.exception) == "captured job", result.output
    assert captured[0].agents[0].extra_allowed_hosts == (
        ["endpoint.test", "existing.test"] if from_config else ["endpoint.test"]
    )


@pytest.mark.parametrize(
    "policy",
    [
        '[environment]\nnetwork_mode = "no-network"',
        '[agent]\nnetwork_mode = "no-network"',
        "[environment]\nallow_internet = false",
    ],
)
def test_trial_keeps_endpoint_on_restricted_network(monkeypatch, tmp_path, policy):
    from pier.environments.factory import EnvironmentFactory

    (tmp_path / "task.toml").write_text(policy)
    (tmp_path / "instruction.md").write_text("Do nothing.")
    task = Task(tmp_path)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", create)
    TrialExecution._create_environment(
        environment_config=EnvironmentConfig(type="daytona"),
        task=task,
        session_id="test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        logger=logging.getLogger("test"),
        agent=SimpleNamespace(
            install_spec=lambda: None,
            network_allowlist=lambda: NetworkAllowlist(domains=["native.test"]),
        ),
        agent_config=AgentConfig(extra_allowed_hosts=["endpoint.test"]),
    )
    assert captured["network_allowlist"].domains == ["endpoint.test", "native.test"]
    env = DaytonaEnvironment.__new__(DaytonaEnvironment)
    env._explicit_network_allow_list = None
    env._explicit_network_block_all = None
    env._resolved_network_allow_list = None
    env._network_resolution_debug = {}
    env.task_env_config = captured["task_env_config"]
    env.network_allowlist = captured["network_allowlist"]
    env.logger = logging.getLogger("test")

    def resolve(domains):
        assert "endpoint.test" in domains
        return {"endpoint.test": ["203.0.113.10"]}, ["203.0.113.10/32"]

    monkeypatch.setattr(
        "pier.environments.daytona.resolve_network_allowlist_to_daytona_cidrs", resolve
    )
    assert env._network_params() == {
        "network_block_all": False,
        "network_allow_list": "203.0.113.10/32",
    }


def test_agent_hosts_reject_urls():
    with pytest.raises(ValueError):
        AgentConfig(extra_allowed_hosts=["https://endpoint.test/v1"])
