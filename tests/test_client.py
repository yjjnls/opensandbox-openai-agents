"""Client lifecycle tests: create, resume, delete, and provider error mapping."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from opensandbox.exceptions import SandboxApiException

from agents.sandbox import Manifest
from agents.sandbox.entries import (
    InContainerMountStrategy,
    MountpointMountPattern,
    S3Mount,
)
from agents.sandbox.errors import MountConfigError, WorkspaceStartError
from agents.sandbox.session import SandboxSession, SandboxSessionState
from agents.sandbox.snapshot import NoopSnapshot

import fakes
from opensandbox_agents import (
    OpenSandboxClient,
    OpenSandboxClientOptions,
    OpenSandboxSession,
    OpenSandboxSessionState,
)
from opensandbox_agents.sandbox import _reject_unsupported_mounts


def make_client() -> OpenSandboxClient:
    return OpenSandboxClient(api_key="key:secret", domain="sandbox.example.com")


def make_options(**overrides) -> OpenSandboxClientOptions:
    payload: dict = {"image": "img:1"}
    payload.update(overrides)
    return OpenSandboxClientOptions(**payload)


def mount_manifest() -> Manifest:
    return Manifest(
        root="/workspace",
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=MountpointMountPattern()),
            ),
        },
    )


class TestCreate:
    async def test_create_builds_state_from_options(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox("sbx-created")
        options = make_options(
            env_vars={"FOO": "bar"},
            resource={"cpu": "1"},
            entrypoint=("sleep", "infinity"),
            sandbox_ttl_s=1800,
            metadata={"purpose": "test"},
        )
        with patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)) as create:
            session = await client.create(manifest=Manifest(root="/workspace"), options=options)

        create.assert_awaited_once()
        kwargs = create.await_args.kwargs
        assert kwargs["image"] == "img:1"
        assert kwargs["env"] == {"FOO": "bar"}
        assert kwargs["resource"] == {"cpu": "1"}
        assert kwargs["entrypoint"] == ["sleep", "infinity"]
        assert kwargs["metadata"] == {"purpose": "test"}
        assert kwargs["timeout"] == timedelta(seconds=1800)
        assert kwargs["connection_config"] is client._connection_config

        assert isinstance(session, SandboxSession)
        inner = session._inner
        assert isinstance(inner, OpenSandboxSession)
        assert inner.sandbox_id == "sbx-created"
        assert inner.state.base_env_vars == {"FOO": "bar"}
        assert inner.state.sandbox_ttl_s == 1800

    async def test_create_wraps_provider_failure(self) -> None:
        client = make_client()
        with patch(
            "opensandbox.Sandbox.create",
            new=AsyncMock(side_effect=SandboxApiException(message="quota", status_code=429)),
        ):
            with pytest.raises(WorkspaceStartError):
                await client.create(options=make_options())

    async def test_create_requires_creation_source_before_allocation(self) -> None:
        client = make_client()
        with patch("opensandbox.Sandbox.create", new=AsyncMock()) as create:
            with pytest.raises(ValueError, match="exactly one"):
                await client.create(options=OpenSandboxClientOptions())
            with pytest.raises(ValueError, match="exactly one"):
                await client.create(
                    options=OpenSandboxClientOptions(image="a", snapshot_id="b")
                )
            create.assert_not_awaited()

    async def test_create_uses_snapshot_id_without_image(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox("sbx-snap")
        with patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)) as create:
            session = await client.create(
                options=make_options(image=None, snapshot_id="imgsnap-1")
            )
        create.assert_awaited_once()
        assert create.await_args.kwargs["snapshot_id"] == "imgsnap-1"
        assert "image" not in create.await_args.kwargs
        assert session.state.snapshot_id == "imgsnap-1"

    async def test_create_rejects_manifest_mounts_before_allocation(self) -> None:
        client = make_client()
        with patch("opensandbox.Sandbox.create", new=AsyncMock()) as create:
            with pytest.raises(MountConfigError):
                await client.create(manifest=mount_manifest(), options=make_options())
            create.assert_not_awaited()

    async def test_create_returns_instrumented_wrapper(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox()
        with patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)):
            session = await client.create(options=make_options())
        assert isinstance(session, SandboxSession)


class TestResume:
    async def test_resume_reconnects_to_existing_sandbox(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox("sbx-existing")
        state = OpenSandboxSessionState(
            sandbox_id="sbx-existing",
            manifest=Manifest(root="/workspace"),
            snapshot=NoopSnapshot(id="snap"),
            image="img:1",
        )
        with patch("opensandbox.Sandbox.connect", new=AsyncMock(return_value=fake)) as connect:
            session = await client.resume(state)
        connect.assert_awaited_once()
        assert connect.await_args.args[0] == "sbx-existing"
        inner = session._inner
        assert isinstance(inner, OpenSandboxSession)
        assert inner.sandbox_id == "sbx-existing"
        assert inner.state.workspace_root_ready == state.workspace_root_ready

    async def test_resume_recreates_when_connect_fails(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox("sbx-new")
        state = OpenSandboxSessionState(
            sandbox_id="sbx-gone",
            manifest=Manifest(root="/workspace"),
            snapshot=NoopSnapshot(id="snap"),
            image="img:1",
        )
        with (
            patch(
                "opensandbox.Sandbox.connect",
                new=AsyncMock(side_effect=SandboxApiException(message="gone", status_code=404)),
            ),
            patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)) as create,
        ):
            session = await client.resume(state)
        create.assert_awaited_once()
        assert create.await_args.kwargs["image"] == "img:1"
        assert session._inner.sandbox_id == "sbx-new"
        assert state.sandbox_id == "sbx-new"
        assert state.workspace_root_ready is False

    async def test_resume_uses_pause_resume_for_paused_sandboxes(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox("sbx-paused")
        state = OpenSandboxSessionState(
            sandbox_id="sbx-paused",
            manifest=Manifest(root="/workspace"),
            snapshot=NoopSnapshot(id="snap"),
            image="img:1",
            pause_on_exit=True,
        )
        with (
            patch(
                "opensandbox.Sandbox.connect",
                new=AsyncMock(side_effect=SandboxApiException(message="paused")),
            ),
            patch("opensandbox.Sandbox.resume", new=AsyncMock(return_value=fake)) as resume,
        ):
            session = await client.resume(state)
        resume.assert_awaited_once()
        assert resume.await_args.args[0] == "sbx-paused"
        assert session._inner.sandbox_id == "sbx-paused"

    async def test_resume_rejects_state_without_creation_source(self) -> None:
        client = make_client()
        state = OpenSandboxSessionState(
            sandbox_id="sbx-gone",
            manifest=Manifest(root="/workspace"),
            snapshot=NoopSnapshot(id="snap"),
        )
        with patch(
            "opensandbox.Sandbox.connect",
            new=AsyncMock(side_effect=SandboxApiException(message="gone", status_code=404)),
        ):
            with pytest.raises(ValueError, match="exactly one"):
                await client.resume(state)

    async def test_resume_rejects_foreign_state(self) -> None:
        client = make_client()
        with pytest.raises(TypeError):
            await client.resume(
                SandboxSessionState(
                    type="unknown",
                    manifest=Manifest(root="/workspace"),
                    snapshot=NoopSnapshot(id="snap"),
                )
            )


class TestDeleteAndClose:
    async def test_delete_shuts_down_inner_session(self) -> None:
        client = make_client()
        fake = fakes.make_fake_sandbox()
        with patch("opensandbox.Sandbox.create", new=AsyncMock(return_value=fake)):
            session = await client.create(options=make_options())
        returned = await client.delete(session)
        assert returned is session
        assert fake.kill_calls == 1

    async def test_delete_rejects_foreign_sessions(self) -> None:
        client = make_client()

        class _Foreign:
            _inner = object()

        with pytest.raises(TypeError):
            await client.delete(_Foreign())  # type: ignore[arg-type]

    async def test_close_is_safe_and_reentrant(self) -> None:
        client = make_client()
        await client.close()
        await client.close()


class TestMountRejection:
    def test_reject_unsupported_mounts_reports_backend(self) -> None:
        with pytest.raises(MountConfigError, match="does not support storage mounts"):
            _reject_unsupported_mounts(mount_manifest())

    def test_plain_manifest_passes(self) -> None:
        _reject_unsupported_mounts(Manifest(root="/workspace"))
