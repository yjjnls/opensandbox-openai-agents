"""
OpenSandbox (https://github.com/opensandbox-group/OpenSandbox) sandbox provider for
the OpenAI Agents SDK.

This module implements a sandbox client/session pair backed by the OpenSandbox
Python SDK (`opensandbox`). Applications construct an `OpenSandboxClient` with their
own connection configuration and pass it to `SandboxRunConfig(client=...)`; the SDK
never imports this package itself.

The `opensandbox` dependency is declared as a hard package dependency, but imports
stay lazy so importing this package never touches the network or starts transports.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import shlex
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from agents.sandbox import Manifest
from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    MountConfigError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from agents.sandbox.entries import Mount
from agents.sandbox.session import (
    BaseSandboxClient,
    Instrumentation,
    SandboxSession,
    SandboxSessionState,
)
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER
from agents.sandbox.session.sandbox_client import BaseSandboxClientOptions as OptionsBase
from agents.sandbox.session.tar_workspace import shell_tar_exclude_args
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from agents.sandbox.types import ExecResult, ExposedPortEndpoint, User
from agents.sandbox.util.retry import (
    TRANSIENT_HTTP_STATUS_CODES,
    exception_chain_contains_type,
    exception_chain_has_status_code,
    iter_exception_chain,
    retry_async,
)
from agents.sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from agents.sandbox.workspace_paths import (
    coerce_posix_path,
    posix_path_as_path,
    posix_path_for_error,
    sandbox_path_str,
)

logger = logging.getLogger(__name__)

DEFAULT_OPENSANDBOX_WORKSPACE_ROOT = "/workspace"

# OpenSandbox API failures carry an HTTP status code on `SandboxApiException` when the
# response completed. The retryability defaults mirror the SDK's transient-status set.
_OPENSANDBOX_HTTP_STATUS_RETRYABLE: dict[int, bool] = {
    400: False,
    401: False,
    403: False,
    404: False,
    409: False,
    429: True,
    500: True,
    502: True,
    503: True,
    504: True,
}


def _import_opensandbox_sdk() -> dict[str, Any]:
    """Lazily import OpenSandbox SDK symbols, raising a clear error if missing."""
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.exceptions import (
            InvalidArgumentException,
            SandboxApiException,
            SandboxConnectionException,
            SandboxException,
            SandboxRateLimitException,
            SandboxReadyTimeoutException,
            SandboxTimeoutException,
            SandboxUnhealthyException,
        )
        from opensandbox.models.execd import RunCommandOpts
    except ImportError as e:
        raise ImportError(
            "OpenSandboxClient requires the `opensandbox` dependency.\n"
            "Install this package (opensandbox-openai-agents) so the OpenSandbox "
            "Python SDK is available before using this sandbox backend."
        ) from e
    return {
        "Sandbox": Sandbox,
        "ConnectionConfig": ConnectionConfig,
        "InvalidArgumentException": InvalidArgumentException,
        "SandboxApiException": SandboxApiException,
        "SandboxConnectionException": SandboxConnectionException,
        "SandboxException": SandboxException,
        "SandboxRateLimitException": SandboxRateLimitException,
        "SandboxReadyTimeoutException": SandboxReadyTimeoutException,
        "SandboxTimeoutException": SandboxTimeoutException,
        "SandboxUnhealthyException": SandboxUnhealthyException,
        "RunCommandOpts": RunCommandOpts,
    }


def _opensandbox_exception_types(*names: str) -> tuple[type[BaseException], ...]:
    """Best-effort import of OpenSandbox exception classes by name."""
    try:
        import opensandbox.exceptions as exceptions_module
    except Exception:
        return ()

    collected: list[type[BaseException]] = []
    for name in names:
        value = getattr(exceptions_module, name, None)
        if isinstance(value, type) and issubclass(value, BaseException):
            collected.append(value)
    return tuple(collected)


def _opensandbox_timeout_error_types() -> tuple[type[BaseException], ...]:
    return _opensandbox_exception_types(
        "SandboxTimeoutException",
        "SandboxReadyTimeoutException",
        "SandboxWakeTimeoutException",
    )


def _opensandbox_retryable_error_types() -> tuple[type[BaseException], ...]:
    return _opensandbox_exception_types(
        "SandboxRateLimitException",
        "SandboxConnectionException",
        "SandboxUnhealthyException",
    )


def _opensandbox_not_found_error_types() -> tuple[type[BaseException], ...]:
    return _opensandbox_exception_types("SandboxNotFoundException")


def _retryable_persist_workspace_error_types() -> tuple[type[BaseException], ...]:
    return (asyncio.TimeoutError, *_opensandbox_timeout_error_types())


def _opensandbox_provider_error_detail(error: BaseException) -> str | None:
    message = str(error)
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        if message:
            return f"HTTP {status}: {message}"
        return f"HTTP {status}"
    request_id = getattr(error, "request_id", None)
    suffix = f" (request_id={request_id})" if request_id else ""
    if message:
        return f"{type(error).__name__}: {message}{suffix}"
    return type(error).__name__ + suffix


def _opensandbox_provider_retryability(error: BaseException) -> tuple[bool | None, str | None]:
    """Classify a provider error as retryable or not, with a short reason code."""
    timeout_types = _opensandbox_timeout_error_types()
    if timeout_types and isinstance(error, timeout_types):
        return False, "provider_timeout"

    retryable_types = _opensandbox_retryable_error_types()
    if retryable_types and isinstance(error, retryable_types):
        return True, type(error).__name__

    not_found_types = _opensandbox_not_found_error_types()
    if not_found_types and isinstance(error, not_found_types):
        return False, "sandbox_not_found"

    for candidate in iter_exception_chain(error):
        sdk_retryable = getattr(candidate, "is_retryable", None)
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int):
            mapped = _OPENSANDBOX_HTTP_STATUS_RETRYABLE.get(status)
            if mapped is not None:
                return mapped, f"http_{status}"
        if isinstance(sdk_retryable, bool):
            return sdk_retryable, "sdk_classification"
    return None, None


class OpenSandboxTimeouts(BaseModel):
    """Timeout configuration for OpenSandbox provider operations, in seconds."""

    model_config = {"frozen": True}

    exec_timeout_unbounded_s: int = Field(default=24 * 60 * 60, ge=1)
    keepalive_s: int = Field(default=10, ge=1)
    cleanup_s: int = Field(default=30, ge=1)
    fast_op_s: int = Field(default=30, ge=1)
    file_upload_s: int = Field(default=1800, ge=1)
    file_download_s: int = Field(default=1800, ge=1)
    workspace_tar_s: int = Field(default=300, ge=1)


class OpenSandboxClientOptions(OptionsBase):
    """Client options for the OpenSandbox sandbox provider.

    Exactly one of `image` or `snapshot_id` must be set. Credentials and the
    connection configuration live on the `OpenSandboxClient`, never here: these
    options are serializable and must stay free of secrets.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["opensandbox"] = "opensandbox"
    image: str | None = Field(default=None, min_length=1)
    snapshot_id: str | None = Field(default=None, min_length=1)
    workspace_root: str = DEFAULT_OPENSANDBOX_WORKSPACE_ROOT
    env_vars: dict[str, str] | None = None
    metadata: dict[str, str] | None = None
    resource: dict[str, str] | None = None
    entrypoint: tuple[str, ...] | None = None
    sandbox_ttl_s: int | None = Field(default=None, ge=1)
    ready_timeout_s: int = Field(default=60, ge=1)
    pause_on_exit: bool = False
    timeouts: OpenSandboxTimeouts | dict[str, object] | None = None
    exposed_ports: tuple[int, ...] = ()

    def __init__(
        self,
        image: str | None = None,
        snapshot_id: str | None = None,
        workspace_root: str = DEFAULT_OPENSANDBOX_WORKSPACE_ROOT,
        env_vars: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        resource: dict[str, str] | None = None,
        entrypoint: tuple[str, ...] | None = None,
        sandbox_ttl_s: int | None = None,
        ready_timeout_s: int = 60,
        pause_on_exit: bool = False,
        timeouts: OpenSandboxTimeouts | dict[str, object] | None = None,
        exposed_ports: tuple[int, ...] = (),
        *,
        type: Literal["opensandbox"] = "opensandbox",
    ) -> None:
        super().__init__(
            type=type,
            image=image,
            snapshot_id=snapshot_id,
            workspace_root=workspace_root,
            env_vars=env_vars,
            metadata=metadata,
            resource=resource,
            entrypoint=entrypoint,
            sandbox_ttl_s=sandbox_ttl_s,
            ready_timeout_s=ready_timeout_s,
            pause_on_exit=pause_on_exit,
            timeouts=timeouts,
            exposed_ports=exposed_ports,
        )

    def resolve_timeouts(self) -> OpenSandboxTimeouts:
        timeouts_in = self.timeouts
        if isinstance(timeouts_in, OpenSandboxTimeouts):
            return timeouts_in
        if timeouts_in is None:
            return OpenSandboxTimeouts()
        return OpenSandboxTimeouts.model_validate(timeouts_in)

    def validate_creation_source(self) -> None:
        if (self.image is None) == (self.snapshot_id is None):
            raise ValueError(
                "OpenSandboxClientOptions requires exactly one of `image` or "
                "`snapshot_id` to create a sandbox."
            )


class OpenSandboxSessionState(SandboxSessionState):
    """Serializable state for an OpenSandbox-backed session."""

    type: Literal["opensandbox"] = "opensandbox"
    sandbox_id: str
    image: str | None = None
    snapshot_id: str | None = None
    workspace_root: str = DEFAULT_OPENSANDBOX_WORKSPACE_ROOT
    base_env_vars: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resource: dict[str, str] = Field(default_factory=dict)
    entrypoint: tuple[str, ...] | None = None
    sandbox_ttl_s: int | None = None
    ready_timeout_s: int = 60
    pause_on_exit: bool = False
    timeouts: OpenSandboxTimeouts = Field(default_factory=OpenSandboxTimeouts)


class OpenSandboxSession(BaseSandboxSession):
    """OpenSandbox-backed sandbox session implementation."""

    state: OpenSandboxSessionState
    _sandbox: Any

    def __init__(self, *, state: OpenSandboxSessionState, sandbox: Any) -> None:
        self.state = state
        self._sandbox = sandbox

    @classmethod
    def from_state(
        cls,
        state: OpenSandboxSessionState,
        *,
        sandbox: Any,
    ) -> OpenSandboxSession:
        return cls(state=state, sandbox=sandbox)

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            endpoint = await self._sandbox.get_endpoint(port)
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "opensandbox", "detail": "get_endpoint_failed"},
                cause=e,
            ) from e

        url = getattr(endpoint, "endpoint", None)
        if not isinstance(url, str) or not url:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "opensandbox", "detail": "invalid_endpoint", "url": url},
            )

        try:
            split = urlsplit(url)
            host = split.hostname
            if host is None:
                raise ValueError("missing hostname")
            port_value = split.port or (443 if split.scheme == "https" else 80)
            return ExposedPortEndpoint(host=host, port=port_value, tls=split.scheme == "https")
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "opensandbox", "detail": "invalid_endpoint", "url": url},
                cause=e,
            ) from e

    async def _shutdown_backend(self) -> None:
        try:
            if self.state.pause_on_exit:
                await self._sandbox.pause()
            else:
                await self._sandbox.kill()
        except Exception:
            pass
        finally:
            try:
                await self._sandbox.close()
            except Exception:
                pass

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    def _runtime_helpers(self) -> tuple[Any, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _resolved_envs(self) -> dict[str, str]:
        manifest_envs = await self.state.manifest.environment.resolve()
        return {**self.state.base_env_vars, **manifest_envs}

    def _coerce_exec_timeout(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return float(self.state.timeouts.exec_timeout_unbounded_s)
        if timeout_s <= 0:
            return 0.001
        return float(timeout_s)

    def _run_command_opts(self, cwd: str, envs: dict[str, str], timeout_s: float) -> Any:
        RunCommandOpts = _import_opensandbox_sdk()["RunCommandOpts"]
        return RunCommandOpts(
            working_directory=cwd,
            envs=envs or None,
            timeout=timedelta(seconds=math.ceil(timeout_s)),
        )

    async def _prepare_backend_workspace(self) -> None:
        """Create the workspace root before SDK exec calls use it as cwd."""
        root = sandbox_path_str(self.state.manifest.root)
        error_root = posix_path_for_error(root)
        envs = await self._resolved_envs()
        try:
            result = await self._sandbox.commands.run(
                f"mkdir -p -- {shlex.quote(root)}",
                opts=self._run_command_opts("/", envs, float(self.state.timeouts.fast_op_s)),
            )
        except Exception as e:
            detail = _opensandbox_provider_error_detail(e)
            message = "failed to start session"
            if detail:
                message = f"{message}: OpenSandbox workspace root setup failed: {detail}"
            raise WorkspaceStartError(
                path=error_root,
                context={"backend": "opensandbox", "reason": "workspace_root_setup_failed"},
                cause=e,
                message=message,
            ) from e

        exit_code = int(getattr(result, "exit_code", 0) or 0)
        if exit_code != 0:
            output = _execution_output(result)
            message = (
                f"failed to start session: OpenSandbox workspace root setup exited "
                f"with {exit_code}"
            )
            if output:
                message = f"{message}: {output}"
            raise WorkspaceStartError(
                path=error_root,
                context={
                    "backend": "opensandbox",
                    "reason": "workspace_root_nonzero_exit",
                    "exit_code": exit_code,
                    "output": output,
                },
                message=message,
            )

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        cmd_str = shlex.join(str(c) for c in command)
        envs = await self._resolved_envs()
        cwd = sandbox_path_str(self.state.manifest.root)

        caller_timeout = self._coerce_exec_timeout(timeout)
        deadline = time.monotonic() + caller_timeout
        timeout_error_types = _opensandbox_timeout_error_types()

        def _remaining_timeout() -> float:
            return max(0.001, deadline - time.monotonic())

        try:
            result = await asyncio.wait_for(
                self._sandbox.commands.run(
                    cmd_str,
                    opts=self._run_command_opts(cwd, envs, _remaining_timeout()),
                ),
                timeout=caller_timeout,
            )
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            stdout, stderr = _execution_streams(result, exit_code)
            return ExecResult(
                stdout=stdout.encode("utf-8", errors="replace"),
                stderr=stderr.encode("utf-8", errors="replace"),
                exit_code=exit_code,
            )
        except asyncio.TimeoutError as e:
            raise ExecTimeoutError(command=command, timeout_s=timeout, cause=e) from e
        except Exception as e:
            if isinstance(e, timeout_error_types):
                raise ExecTimeoutError(command=command, timeout_s=timeout, cause=e) from e
            raise _opensandbox_exec_transport_error(command=command, cause=e) from e

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)

        not_found_error_types = _opensandbox_not_found_error_types()

        try:
            data: bytes = await self._sandbox.files.read_bytes(sandbox_path_str(workspace_path))
            return io.BytesIO(data)
        except Exception as e:
            if _is_not_found(e, not_found_error_types):
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            retryable, _ = _opensandbox_provider_retryability(e)
            raise WorkspaceArchiveReadError(path=error_path, cause=e, retryable=retryable) from e

    async def write(
        self,
        path: Path | str,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)

        if not hasattr(data, "read"):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(data).__name__)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)

        workspace_path = await self._validate_path_access(path, for_write=True)
        try:
            await self._sandbox.files.write_file(
                sandbox_path_str(workspace_path),
                bytes(payload),
            )
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=workspace_path, cause=e) from e

    async def running(self) -> bool:
        try:
            return bool(
                await asyncio.wait_for(
                    self._sandbox.is_healthy(),
                    timeout=float(self.state.timeouts.keepalive_s),
                )
            )
        except Exception:
            return False

    def _tar_exclude_args(self) -> list[str]:
        return shell_tar_exclude_args(self._persist_workspace_skip_relpaths())

    @retry_async(
        retry_if=lambda exc, self, tar_cmd, tar_path: (
            exception_chain_contains_type(exc, _retryable_persist_workspace_error_types())
            or exception_chain_has_status_code(exc, TRANSIENT_HTTP_STATUS_CODES)
        )
    )
    async def _run_persist_workspace_command(self, tar_cmd: str, tar_path: str) -> bytes:
        try:
            envs = await self._resolved_envs()
            result = await self._sandbox.commands.run(
                tar_cmd,
                opts=self._run_command_opts(
                    "/", envs, float(self.state.timeouts.workspace_tar_s)
                ),
            )
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            if exit_code != 0:
                raise WorkspaceArchiveReadError(
                    path=self._workspace_root_path(),
                    context={
                        "reason": "tar_failed",
                        "output": _execution_output(result),
                    },
                    retryable=False,
                )
            return cast(
                bytes,
                await self._sandbox.files.read_bytes(tar_path),
            )
        except WorkspaceArchiveReadError:
            raise
        except Exception as e:
            detail = _opensandbox_provider_error_detail(e)
            retryable, reason = _opensandbox_provider_retryability(e)
            context: dict[str, object] = {"backend": "opensandbox"}
            if reason is not None:
                context["reason"] = reason
            if detail:
                context["provider_error"] = detail
            request_id = getattr(e, "request_id", None)
            if request_id:
                context["request_id"] = request_id
            raise WorkspaceArchiveReadError(
                path=self._workspace_root_path(),
                context=context,
                cause=e,
                retryable=retryable,
            ) from e

    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        tar_path = f"/tmp/sandbox-persist-{self.state.session_id.hex}.tar"
        excludes = " ".join(self._tar_exclude_args())
        tar_cmd = (
            f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        ).strip()

        snapshot_error: WorkspaceArchiveReadError | None = None
        raw: bytes | None = None
        try:
            raw = await self._run_persist_workspace_command(tar_cmd, tar_path)
        except WorkspaceArchiveReadError as e:
            snapshot_error = e
        finally:
            try:
                await self._sandbox.files.delete_files([tar_path])
            except Exception:
                pass

        if snapshot_error is not None:
            raise snapshot_error

        assert raw is not None
        return io.BytesIO(raw)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/sandbox-hydrate-{self.state.session_id.hex}.tar"
        if not hasattr(data, "read"):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(data).__name__)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)

        try:
            validate_tar_bytes(
                bytes(payload),
                allow_external_symlink_targets=False,
            )
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={
                    "reason": "unsafe_or_invalid_tar",
                    "member": e.member,
                    "detail": str(e),
                },
                cause=e,
            ) from e

        try:
            envs = await self._resolved_envs()
            await self._sandbox.files.write_file(tar_path, bytes(payload))
            result = await self._sandbox.commands.run(
                f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                opts=self._run_command_opts(
                    "/", envs, float(self.state.timeouts.workspace_tar_s)
                ),
            )
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            if exit_code != 0:
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "reason": "tar_extract_failed",
                        "output": _execution_output(result),
                    },
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=root, cause=e) from e
        finally:
            try:
                await self._sandbox.files.delete_files([tar_path])
            except Exception:
                pass


def _execution_streams(result: Any, exit_code: int) -> tuple[str, str]:
    """Extract stdout/stderr text from an OpenSandbox Execution object."""
    logs = getattr(result, "logs", None)
    stdout_text = _output_messages_text(getattr(logs, "stdout", None))
    stderr_text = _output_messages_text(getattr(logs, "stderr", None))
    if stdout_text or stderr_text:
        return stdout_text, stderr_text
    output = getattr(result, "result", "") or ""
    if exit_code == 0:
        return output, ""
    return "", output


def _output_messages_text(messages: Any) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _execution_output(result: Any) -> str:
    stdout, stderr = _execution_streams(result, 0)
    return stdout or stderr


def _is_not_found(error: BaseException, not_found_error_types: tuple[type[BaseException], ...]) -> bool:
    if not_found_error_types and isinstance(error, not_found_error_types):
        return True
    for candidate in iter_exception_chain(error):
        status = getattr(candidate, "status_code", None)
        if status == 404:
            return True
    return False


def _opensandbox_exec_transport_error(
    *,
    command: tuple[str | Path, ...],
    cause: BaseException,
) -> ExecTransportError:
    detail = _opensandbox_provider_error_detail(cause)
    context: dict[str, object] = {"backend": "opensandbox"}
    retryable, reason = _opensandbox_provider_retryability(cause)
    if reason is not None:
        context["reason"] = reason
    if detail:
        context["provider_error"] = detail
    status = getattr(cause, "status_code", None)
    if isinstance(status, int):
        context["http_status"] = status
    request_id = getattr(cause, "request_id", None)
    if request_id:
        context["request_id"] = request_id
    message = "OpenSandbox exec failed"
    if detail:
        message = f"{message}: {detail}"
    return ExecTransportError(
        command=command,
        context=context,
        cause=cause,
        message=message,
        retryable=retryable,
    )


class OpenSandboxClient(BaseSandboxClient[OpenSandboxClientOptions]):
    """OpenSandbox sandbox client managing sandbox lifecycle via the OpenSandbox SDK.

    The client owns the `ConnectionConfig` (endpoint and credentials). Applications
    close the client when finished; closing it does not stop sandboxes owned by
    still-open sessions.
    """

    backend_id = "opensandbox"
    _instrumentation: Instrumentation

    def __init__(
        self,
        *,
        connection_config: Any | None = None,
        domain: str | None = None,
        api_key: str | None = None,
        protocol: str | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Any | None = None,
    ) -> None:
        sdk = _import_opensandbox_sdk()
        if connection_config is None:
            kwargs: dict[str, object] = {}
            if domain is not None:
                kwargs["domain"] = domain
            if api_key is not None:
                kwargs["api_key"] = api_key
            if protocol is not None:
                kwargs["protocol"] = protocol
            connection_config = sdk["ConnectionConfig"](**kwargs)
        self._connection_config = connection_config
        self._instrumentation = (
            instrumentation if instrumentation is not None else Instrumentation()
        )
        self._dependencies = dependencies

    def _creation_params(
        self,
        *,
        image: str | None,
        snapshot_id: str | None,
        env_vars: dict[str, str] | None,
        sandbox_ttl_s: int | None,
        ready_timeout_s: int,
        resource: dict[str, str] | None,
        entrypoint: tuple[str, ...] | None,
        metadata: dict[str, str] | None,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "env": dict(env_vars) if env_vars else None,
            "timeout": timedelta(seconds=sandbox_ttl_s) if sandbox_ttl_s else None,
            "ready_timeout": timedelta(seconds=ready_timeout_s),
            "resource": dict(resource) if resource else None,
            "entrypoint": list(entrypoint) if entrypoint else None,
            "metadata": dict(metadata) if metadata else None,
            "connection_config": self._connection_config,
        }
        if snapshot_id is not None:
            params["snapshot_id"] = snapshot_id
        else:
            params["image"] = image
        return params

    @staticmethod
    def _reject_invalid_creation_options(
        image: str | None, snapshot_id: str | None
    ) -> None:
        if (image is None) == (snapshot_id is None):
            raise ValueError(
                "OpenSandboxClientOptions requires exactly one of `image` or "
                "`snapshot_id` to create a sandbox."
            )

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: OpenSandboxClientOptions,
    ) -> SandboxSession:
        if not isinstance(options, OpenSandboxClientOptions):
            raise TypeError(
                "OpenSandboxClient.create expects an OpenSandboxClientOptions instance"
            )
        self._reject_invalid_creation_options(options.image, options.snapshot_id)

        if manifest is None:
            manifest = Manifest(root=options.workspace_root)
        self._validate_manifest_for_create(manifest)
        _reject_unsupported_mounts(manifest)

        timeouts = options.resolve_timeouts()
        Sandbox = _import_opensandbox_sdk()["Sandbox"]

        session_id = uuid.uuid4()
        try:
            opensandbox_sandbox = await Sandbox.create(
                **self._creation_params(
                    image=options.image,
                    snapshot_id=options.snapshot_id,
                    env_vars=options.env_vars,
                    sandbox_ttl_s=options.sandbox_ttl_s,
                    ready_timeout_s=options.ready_timeout_s,
                    resource=options.resource,
                    entrypoint=options.entrypoint,
                    metadata=options.metadata,
                )
            )
        except Exception as e:
            detail = _opensandbox_provider_error_detail(e)
            message = "failed to create OpenSandbox sandbox"
            if detail:
                message = f"{message}: {detail}"
            raise WorkspaceStartError(
                path=posix_path_for_error(options.workspace_root),
                context={"backend": "opensandbox", "reason": "sandbox_create_failed"},
                cause=e,
                message=message,
            ) from e

        snapshot_instance = resolve_snapshot(snapshot, str(session_id))
        state = OpenSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=snapshot_instance,
            sandbox_id=_sandbox_instance_id(opensandbox_sandbox),
            image=options.image,
            snapshot_id=options.snapshot_id,
            workspace_root=options.workspace_root,
            base_env_vars=dict(options.env_vars or {}),
            metadata=dict(options.metadata or {}),
            resource=dict(options.resource or {}),
            entrypoint=options.entrypoint,
            sandbox_ttl_s=options.sandbox_ttl_s,
            ready_timeout_s=options.ready_timeout_s,
            pause_on_exit=options.pause_on_exit,
            timeouts=timeouts,
            exposed_ports=options.exposed_ports,
        )
        inner = OpenSandboxSession.from_state(state, sandbox=opensandbox_sandbox)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def close(self) -> None:
        """Release client-held resources.

        The OpenSandbox SDK attaches per-sandbox transports to each `Sandbox`
        instance, so there is no client-level transport to close. Running
        sandboxes are unaffected; they are torn down through their sessions.
        """

    async def __aenter__(self) -> OpenSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, OpenSandboxSession):
            raise TypeError("OpenSandboxClient.delete expects an OpenSandboxSession")
        try:
            await inner.shutdown()
        except Exception:
            pass
        return session

    async def resume(
        self,
        state: SandboxSessionState,
    ) -> SandboxSession:
        if not isinstance(state, OpenSandboxSessionState):
            raise TypeError("OpenSandboxClient.resume expects an OpenSandboxSessionState")
        state.assert_path_grants_rebound()

        sdk = _import_opensandbox_sdk()
        Sandbox = sdk["Sandbox"]
        connect_timeout = timedelta(seconds=max(1, int(state.timeouts.fast_op_s)))

        opensandbox_sandbox = None
        reconnected = False
        try:
            opensandbox_sandbox = await Sandbox.connect(
                state.sandbox_id,
                connection_config=self._connection_config,
                connect_timeout=connect_timeout,
            )
            reconnected = True
        except Exception as connect_error:
            if state.pause_on_exit:
                try:
                    opensandbox_sandbox = await Sandbox.resume(
                        state.sandbox_id,
                        connection_config=self._connection_config,
                    )
                    reconnected = True
                except Exception as resume_error:
                    log_reconnect_failure(resume_error)
            else:
                log_reconnect_failure(connect_error)

        if not reconnected or opensandbox_sandbox is None:
            self._reject_invalid_creation_options(state.image, state.snapshot_id)
            try:
                opensandbox_sandbox = await Sandbox.create(
                    **self._creation_params(
                        image=state.image,
                        snapshot_id=state.snapshot_id,
                        env_vars=state.base_env_vars,
                        sandbox_ttl_s=state.sandbox_ttl_s,
                        ready_timeout_s=state.ready_timeout_s,
                        resource=state.resource or None,
                        entrypoint=state.entrypoint,
                        metadata=state.metadata or None,
                    )
                )
            except Exception as e:
                detail = _opensandbox_provider_error_detail(e)
                message = "failed to recreate OpenSandbox sandbox during resume"
                if detail:
                    message = f"{message}: {detail}"
                raise WorkspaceStartError(
                    path=posix_path_for_error(state.workspace_root),
                    context={"backend": "opensandbox", "reason": "sandbox_recreate_failed"},
                    cause=e,
                    message=message,
                ) from e
            state.sandbox_id = _sandbox_instance_id(opensandbox_sandbox)
            state.workspace_root_ready = False

        inner = OpenSandboxSession.from_state(state, sandbox=opensandbox_sandbox)
        inner._set_start_state_preserved(reconnected, system=reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, OpenSandboxSessionState)


def log_reconnect_failure(error: BaseException | None) -> None:
    if error is None:
        return
    logger.debug(
        "OpenSandbox sandbox reconnect failed; recreating from creation inputs: %s",
        error,
    )


def _sandbox_instance_id(sandbox: Any) -> str:
    """Read the backend sandbox id from an OpenSandbox SDK `Sandbox` instance."""
    value = getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)
    return str(value or "")


def _reject_unsupported_mounts(manifest: Manifest) -> None:
    for entry, _path in manifest.mount_targets():
        if isinstance(entry, Mount):
            raise MountConfigError(
                message=(
                    "OpenSandbox provider does not support storage mounts; remove "
                    f"the mount at `{_path}` or choose a backend with mount support."
                ),
                context={"backend": "opensandbox"},
            )


__all__ = [
    "DEFAULT_OPENSANDBOX_WORKSPACE_ROOT",
    "OpenSandboxClient",
    "OpenSandboxClientOptions",
    "OpenSandboxSession",
    "OpenSandboxSessionState",
    "OpenSandboxTimeouts",
]
