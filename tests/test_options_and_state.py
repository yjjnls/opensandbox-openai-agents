"""Options parsing, type registration, and session-state round-trip tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.sandbox import Manifest
from agents.sandbox.snapshot import NoopSnapshot
from agents.sandbox.session import BaseSandboxClientOptions, SandboxSessionState

import opensandbox_agents
from opensandbox_agents import (
    OpenSandboxClientOptions,
    OpenSandboxSessionState,
    OpenSandboxTimeouts,
)


def test_import_registers_option_and_state_types() -> None:
    options = BaseSandboxClientOptions.parse(
        {"type": "opensandbox", "image": "opensandbox/code-interpreter:latest"}
    )
    assert isinstance(options, OpenSandboxClientOptions)
    assert options.image == "opensandbox/code-interpreter:latest"
    assert options.workspace_root == "/workspace"


def test_options_reject_unknown_fields() -> None:
    # Both dictionary payloads (the SandboxRunConfig parsing path) and direct keyword
    # construction reject unknown field names explicitly.
    with pytest.raises(TypeError):
        OpenSandboxClientOptions.model_validate(
            {"type": "opensandbox", "image": "img:1", "network_mode": "none"}
        )
    with pytest.raises(TypeError):
        OpenSandboxClientOptions(
            image="opensandbox/code-interpreter:latest", network_mode="none"
        )


def test_options_reject_empty_image() -> None:
    with pytest.raises(ValidationError):
        OpenSandboxClientOptions.model_validate({"type": "opensandbox", "image": ""})
    with pytest.raises(ValidationError):
        OpenSandboxClientOptions(image="")


def test_options_reject_non_positive_ttl() -> None:
    with pytest.raises(ValidationError):
        OpenSandboxClientOptions(
            image="opensandbox/code-interpreter:latest", sandbox_ttl_s=0
        )


def test_options_require_exactly_one_creation_source() -> None:
    options = OpenSandboxClientOptions()
    with pytest.raises(ValueError, match="exactly one"):
        options.validate_creation_source()
    both = OpenSandboxClientOptions(
        image="img:1", snapshot_id="snap-1"
    )
    with pytest.raises(ValueError, match="exactly one"):
        both.validate_creation_source()


def test_timeouts_accept_dictionary_payload() -> None:
    options = OpenSandboxClientOptions(
        image="img:1", timeouts={"fast_op_s": 5, "workspace_tar_s": 11}
    )
    timeouts = options.resolve_timeouts()
    assert isinstance(timeouts, OpenSandboxTimeouts)
    assert timeouts.fast_op_s == 5
    assert timeouts.workspace_tar_s == 11
    assert timeouts.exec_timeout_unbounded_s == 24 * 60 * 60


def test_state_type_registration() -> None:
    assert SandboxSessionState._subclass_registry["opensandbox"] is OpenSandboxSessionState
    assert (
        BaseSandboxClientOptions._subclass_registry["opensandbox"] is OpenSandboxClientOptions
    )


def test_state_json_round_trip() -> None:
    client = _state_round_trip_client()
    state = OpenSandboxSessionState(
        sandbox_id="sbx-42",
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snap"),
        image="img:1",
        base_env_vars={"FOO": "bar"},
        resource={"cpu": "1", "memory": "2Gi"},
        entrypoint=("sleep", "infinity"),
        sandbox_ttl_s=3600,
        pause_on_exit=True,
        exposed_ports=(),
    )
    payload = client.serialize_session_state(state)
    assert payload["type"] == "opensandbox"
    assert payload["sandbox_id"] == "sbx-42"

    restored = client.deserialize_session_state(payload)
    assert isinstance(restored, OpenSandboxSessionState)
    assert restored.sandbox_id == "sbx-42"
    assert restored.session_id == state.session_id
    assert restored.image == "img:1"
    assert restored.base_env_vars == {"FOO": "bar"}
    assert restored.entrypoint == ("sleep", "infinity")
    assert restored.pause_on_exit is True
    assert restored.resource == {"cpu": "1", "memory": "2Gi"}
    assert restored.timeouts.exec_timeout_unbounded_s == 24 * 60 * 60


def test_public_exports() -> None:
    assert opensandbox_agents.__version__ == "0.1.0"
    for name in opensandbox_agents.__all__:
        assert getattr(opensandbox_agents, name) is not None


def _state_round_trip_client():
    from opensandbox_agents import OpenSandboxClient

    return OpenSandboxClient(api_key="key:secret", domain="sandbox.example.com")
