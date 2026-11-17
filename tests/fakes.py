"""Shared fake OpenSandbox SDK objects used by the provider tests.

The fakes implement the subset of the OpenSandbox SDK surface the provider touches,
backed by real SDK models where practical so request/response shapes stay honest.
They also emulate the Agents SDK runtime-helper lifecycle (install/probe/invoke) and
workspace tar creation/extraction so session start, path validation, and snapshot
persistence run end to end without a real sandbox.
"""

from __future__ import annotations

import asyncio
import io
import shlex
import tarfile
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage, RunCommandOpts
from opensandbox.models.sandboxes import SandboxEndpoint

HELPER_INSTALL_PREFIX = "/tmp/openai-agents/bin/"
HELPER_INSTALL_MARKER = "INSTALL_RUNTIME_HELPER_V1"


def _message(text: str) -> OutputMessage:
    return OutputMessage(text=text, timestamp=int(time.time()))


def fake_execution(
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    result: str | None = None,
) -> Any:
    """Build an Execution-like object.

    Real `Execution` models are used when the payload fits their schema; the
    `result` fallback shape uses a SimpleNamespace because the SDK's `result`
    field has a richer type than plain text.
    """

    if result is not None:
        return SimpleNamespace(
            exit_code=exit_code,
            logs=ExecutionLogs(),
            result=result,
        )
    logs = ExecutionLogs()
    if stdout:
        logs.add_stdout(_message(stdout))
    if stderr:
        logs.add_stderr(_message(stderr))
    return Execution(id="exec-1", exit_code=exit_code, logs=logs)


def _tar_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w"):
        pass
    return buffer.getvalue()


@dataclass
class _FakeCommands:
    sandbox: _FakeSandbox
    executions: list[Any] = field(default_factory=list)
    calls: list[tuple[str, RunCommandOpts | None]] = field(default_factory=list)
    error: BaseException | None = None
    delay_s: float = 0.0
    helpers_installed: bool = False

    async def run(
        self,
        command: str | list[str],
        *,
        opts: RunCommandOpts | None = None,
        handlers: Any = None,
    ) -> Any:
        _ = handlers
        joined = command if isinstance(command, str) else shlex.join(command)
        self.calls.append((joined, opts))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.executions:
            execution = self.executions.pop(0)
            self.sandbox.completed_commands.append((joined, opts, execution))
            return execution
        emulated = self._emulate(joined)
        self.sandbox.completed_commands.append((joined, opts, emulated))
        return emulated

    def _emulate(self, joined: str) -> Any:
        # Runtime helper lifecycle: install heredoc, presence probe, invocation.
        if HELPER_INSTALL_MARKER in joined:
            self.helpers_installed = True
            return fake_execution(exit_code=0)
        if joined.startswith("test -x ") and HELPER_INSTALL_PREFIX in joined:
            if self.helpers_installed:
                return fake_execution(exit_code=0)
            return fake_execution(exit_code=1, stderr="not installed")
        parts = shlex.split(joined)
        if parts and parts[0].startswith(HELPER_INSTALL_PREFIX):
            # Helper invocation echoes the resolved workspace path.
            candidate = parts[2] if len(parts) > 2 else ""
            return fake_execution(exit_code=0, stdout=candidate)
        # Workspace tar: materialize an archive the file API can read back.
        if " tar " in f" {joined} " and " -cf " in joined:
            for token_index, token in enumerate(parts):
                if token == "-cf" and token_index + 1 < len(parts):
                    tar_path = parts[token_index + 1]
                    self.sandbox.files.files[tar_path] = _tar_bytes()
                    return fake_execution(exit_code=0)
        if " tar " in f" {joined} " and " -xf " in joined:
            return fake_execution(exit_code=0)
        if joined.startswith("mkdir -p"):
            return fake_execution(exit_code=0)
        return fake_execution(exit_code=0)


@dataclass
class _FakeFilesystem:
    sandbox: _FakeSandbox
    files: dict[str, bytes] = field(default_factory=dict)
    write_calls: list[tuple[str, bytes]] = field(default_factory=list)
    read_calls: list[str] = field(default_factory=list)
    delete_calls: list[list[str]] = field(default_factory=list)
    read_error: BaseException | None = None

    async def read_bytes(self, path: str) -> bytes:
        self.read_calls.append(path)
        if self.read_error is not None:
            raise self.read_error
        if path not in self.files:
            from opensandbox.exceptions import SandboxApiException

            raise SandboxApiException(message="not found", status_code=404)
        return self.files[path]

    async def write_file(
        self,
        path: str,
        data: str | bytes | Any,
        *,
        encoding: str = "utf-8",
        mode: int = 755,
        owner: str | None = None,
        group: str | None = None,
    ) -> None:
        _ = encoding, mode, owner, group
        payload = data.read() if hasattr(data, "read") else data
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.write_calls.append((path, bytes(payload)))
        self.files[path] = bytes(payload)

    async def delete_files(self, paths: list[str]) -> None:
        self.delete_calls.append(list(paths))
        for path in paths:
            self.files.pop(path, None)


class _FakeSandbox:
    """Fake `opensandbox.Sandbox` instance with recorded lifecycle calls."""

    def __init__(self, sandbox_id: str = "sbx-test-1") -> None:
        self.id = sandbox_id
        self.completed_commands: list[tuple[str, Any, Any]] = []
        self.commands = _FakeCommands(self)
        self.files = _FakeFilesystem(self)
        self.health = True
        self.endpoints: dict[int, str] = {}
        self.pause_calls = 0
        self.kill_calls = 0
        self.close_calls = 0
        self.health_error: BaseException | None = None
        self.health_delay_s: float = 0.0

    async def is_healthy(self) -> bool:
        if self.health_delay_s:
            await asyncio.sleep(self.health_delay_s)
        if self.health_error is not None:
            raise self.health_error
        return self.health

    async def get_endpoint(self, port: int) -> SandboxEndpoint:
        url = self.endpoints.get(port)
        if url is None:
            from opensandbox.exceptions import SandboxApiException

            raise SandboxApiException(message="no endpoint", status_code=404)
        return SandboxEndpoint(endpoint=url)

    async def pause(self) -> None:
        self.pause_calls += 1

    async def kill(self) -> None:
        self.kill_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def make_fake_sandbox(sandbox_id: str = "sbx-test-1") -> _FakeSandbox:
    return _FakeSandbox(sandbox_id)
