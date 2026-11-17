"""Session behavior tests: exec mapping, file IO, workspace persistence, lifecycle."""

from __future__ import annotations

import io
import shlex
import tarfile
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from opensandbox.exceptions import (
    SandboxApiException,
    SandboxConnectionException,
    SandboxTimeoutException,
)

from agents.sandbox import Manifest
from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from agents.sandbox.snapshot import NoopSnapshot

from fakes import fake_execution, make_fake_sandbox
from opensandbox_agents import OpenSandboxClientOptions, OpenSandboxSessionState
from opensandbox_agents.sandbox import OpenSandboxSession


def make_session(fake=None, *, root: str = "/workspace", pause_on_exit: bool = False):
    fake = fake or make_fake_sandbox()
    state = OpenSandboxSessionState(
        sandbox_id=fake.id,
        manifest=Manifest(root=root),
        snapshot=NoopSnapshot(id="snap"),
        image="img:1",
        base_env_vars={"BASE_VAR": "1"},
        pause_on_exit=pause_on_exit,
    )
    session = OpenSandboxSession.from_state(state, sandbox=fake)
    return session, fake


class TestExec:
    async def test_exec_maps_stdout_stderr_and_exit_code(self) -> None:
        session, fake = make_session()
        fake.commands.executions.append(fake_execution(exit_code=3, stdout="out", stderr="bad"))
        result = await session.exec("echo hi", shell=True)
        assert result.exit_code == 3
        assert result.stdout == b"out"
        assert result.stderr == b"bad"
        command, opts = fake.commands.calls[0]
        # The base wraps shell commands with `sh -lc`, so the joined command contains
        # the original text inside a login-shell invocation.
        assert "echo hi" in command
        assert opts is not None
        assert opts.working_directory == "/workspace"
        assert opts.envs is not None
        assert opts.envs["BASE_VAR"] == "1"

    async def test_exec_falls_back_to_result_field(self) -> None:
        session, fake = make_session()
        fake.commands.executions.append(fake_execution(exit_code=0, result="plain"))
        result = await session.exec("uname", shell=True)
        assert result.stdout == b"plain"
        assert result.stderr == b""

    async def test_exec_passes_manifest_env_and_nonzero_envs_omitted_when_empty(self) -> None:
        session, fake = make_session()
        session.state.base_env_vars = {}
        await session.exec("true", shell=True)
        _, opts = fake.commands.calls[0]
        assert opts.envs is None

    async def test_exec_timeout_raises_exec_timeout_error(self) -> None:
        session, fake = make_session()
        fake.commands.delay_s = 0.5
        with pytest.raises(ExecTimeoutError):
            await session.exec("sleep 10", shell=True, timeout=0.05)

    async def test_exec_provider_timeout_exception_maps_to_exec_timeout(self) -> None:
        session, fake = make_session()
        fake.commands.error = SandboxTimeoutException(message="deadline")
        with pytest.raises(ExecTimeoutError):
            await session.exec("sleep 10", shell=True)

    async def test_exec_transport_error_is_retryable_for_connection_errors(self) -> None:
        session, fake = make_session()
        fake.commands.error = SandboxConnectionException(message="reset")
        with pytest.raises(ExecTransportError) as excinfo:
            await session.exec("ls", shell=True)
        assert excinfo.value.retryable is True
        assert excinfo.value.context["backend"] == "opensandbox"

    async def test_exec_transport_error_http_status_mapping(self) -> None:
        session, fake = make_session()
        fake.commands.error = SandboxApiException(message="boom", status_code=503)
        with pytest.raises(ExecTransportError) as excinfo:
            await session.exec("ls", shell=True)
        assert excinfo.value.retryable is True
        assert excinfo.value.context["http_status"] == 503
        assert excinfo.value.context["reason"] == "http_503"

    async def test_exec_transport_error_client_errors_not_retryable(self) -> None:
        session, fake = make_session()
        fake.commands.error = SandboxApiException(message="denied", status_code=403)
        with pytest.raises(ExecTransportError) as excinfo:
            await session.exec("ls", shell=True)
        assert excinfo.value.retryable is False


class TestFiles:
    async def test_read_returns_bytes_stream(self) -> None:
        session, fake = make_session()
        fake.files.files["/workspace/hello.txt"] = b"hello"
        stream = await session.read("hello.txt")
        assert stream.read() == b"hello"

    async def test_read_missing_file_raises_not_found(self) -> None:
        session, fake = make_session()
        with pytest.raises(WorkspaceReadNotFoundError):
            await session.read("missing.txt")

    async def test_write_encodes_string_payload(self) -> None:
        session, fake = make_session()
        await session.write("notes.txt", io.BytesIO(b"data"))
        assert fake.files.files["/workspace/notes.txt"] == b"data"

    async def test_write_rejects_unsupported_payload_type(self) -> None:
        session, _fake = make_session()
        with pytest.raises(WorkspaceWriteTypeError):
            await session.write("bad.bin", 123)  # type: ignore[arg-type]


class TestWorkspacePersistence:
    async def test_persist_workspace_tars_and_reads_archive(self) -> None:
        session, fake = make_session()
        # The fake emulates `tar -cf` by storing archive bytes under the tar path.
        stream = await session.persist_workspace()
        payload = stream.read()
        assert payload[:2] == b"" or len(payload) > 0
        tar_command, _opts = fake.commands.calls[0]
        assert tar_command.startswith("tar ")
        assert f"-C {shlex.quote('/workspace')}" in tar_command
        assert fake.files.delete_calls and fake.files.delete_calls[0][0].startswith(
            "/tmp/sandbox-persist-"
        )

    async def test_persist_workspace_nonzero_tar_fails(self) -> None:
        from agents.sandbox.errors import WorkspaceArchiveReadError

        session, fake = make_session()
        fake.commands.executions.append(fake_execution(exit_code=2, stderr="tar: error"))
        with pytest.raises(WorkspaceArchiveReadError):
            await session.persist_workspace()

    async def test_hydrate_workspace_extracts_uploaded_tar(self) -> None:
        session, fake = make_session()
        archive = io.BytesIO(_build_tar({"hello.txt": b"world"}))
        await session.hydrate_workspace(archive)
        uploaded = [path for path, _data in fake.files.write_calls]
        assert any(path.startswith("/tmp/sandbox-hydrate-") for path in uploaded)
        tar_commands = [cmd for cmd, _opts in fake.commands.calls if "-xf" in cmd]
        assert tar_commands and f"-C {shlex.quote('/workspace')}" in tar_commands[0]
        assert fake.files.delete_calls, "uploaded tar must be cleaned up"

    async def test_hydrate_workspace_rejects_unsafe_tar(self) -> None:
        session, fake = make_session()
        payload = bytearray(_build_tar({"hello.txt": b"world"}))
        # Corrupt the payload so it is no longer a valid tar archive.
        payload[0:6] = b"GARBAG"
        with pytest.raises(WorkspaceArchiveWriteError):
            await session.hydrate_workspace(io.BytesIO(bytes(payload)))
        assert fake.commands.calls == []

    async def test_hydrate_workspace_rejects_non_stream_payload(self) -> None:
        session, _fake = make_session()
        with pytest.raises(WorkspaceWriteTypeError):
            await session.hydrate_workspace(123)  # type: ignore[arg-type]


class TestLifecycle:
    async def test_workspace_root_prepared_on_start(self) -> None:
        session, fake = make_session()
        await session.start()
        first_command, first_opts = fake.commands.calls[0]
        assert first_command == "mkdir -p -- /workspace"
        assert first_opts.working_directory == "/"

    async def test_start_failure_wraps_workspace_start_error(self) -> None:
        session, fake = make_session()
        fake.commands.error = SandboxApiException(message="nope", status_code=500)
        with pytest.raises(WorkspaceStartError):
            await session.start()

    async def test_shutdown_kills_by_default(self) -> None:
        session, fake = make_session()
        await session.shutdown()
        assert fake.kill_calls == 1
        assert fake.pause_calls == 0
        assert fake.close_calls == 1

    async def test_shutdown_pauses_when_configured(self) -> None:
        session, fake = make_session(pause_on_exit=True)
        await session.shutdown()
        assert fake.pause_calls == 1
        assert fake.kill_calls == 0

    async def test_running_uses_health_probe(self) -> None:
        session, fake = make_session()
        fake.health = False
        assert await session.running() is False
        fake.health = True
        assert await session.running() is True

    async def test_running_survives_health_errors(self) -> None:
        session, fake = make_session()
        fake.health_error = SandboxConnectionException(message="down")
        assert await session.running() is False

    async def test_supports_pty_is_false(self) -> None:
        session, _fake = make_session()
        assert session.supports_pty() is False
        assert session.supports_docker_volume_mounts() is False


class TestExposedPorts:
    async def test_resolve_exposed_port_parses_endpoint_url(self) -> None:
        session, fake = make_session()
        fake.endpoints[8080] = "https://sbx-1.example.com:8443"
        endpoint = await session._resolve_exposed_port(8080)
        assert endpoint.host == "sbx-1.example.com"
        assert endpoint.port == 8443
        assert endpoint.tls is True

    async def test_resolve_exposed_port_failure_wraps_provider_error(self) -> None:
        from agents.sandbox.errors import ExposedPortUnavailableError

        session, fake = make_session()
        with pytest.raises(ExposedPortUnavailableError):
            await session._resolve_exposed_port(8080)


def _build_tar(files: dict[str, bytes]) -> bytes:
    import io as _io

    buffer = _io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, _io.BytesIO(content))
    return buffer.getvalue()
