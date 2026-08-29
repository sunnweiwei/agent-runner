from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
import shutil
import subprocess
import tempfile

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.utils import default_docker_platform
from harbor.models.task.config import NetworkPolicy
from harbor.utils.container_cache import docker_build_context_hash


class NoNetworkDockerEnvironment(DockerEnvironment):
    """Docker provider for Compose tasks already pinned to ``network_mode: none``.

    Harbor's normal Docker provider implements network policies with an nftables
    sidecar. A fully disconnected container needs no sidecar: Compose's network
    namespace is enough, and also works on hosts without a default Docker bridge.
    The staged task is the enforcement point; this provider only advertises that
    enforcement to Harbor's fail-closed capability checks.
    """

    @staticmethod
    def _requires_egress_control(
        *,
        startup_network_policy: NetworkPolicy,
        phase_network_policies: Sequence[NetworkPolicy],
    ) -> bool:
        return False

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            windows=False,
            mounted=True,
            docker_compose=True,
        )


class AllowlistDockerEnvironment(DockerEnvironment):
    """Use Harbor allowlists when Docker's legacy default bridge is absent.

    Harbor's kernel probe needs no network, but upstream runs it on Docker's
    default ``bridge``. Some shared GPU hosts keep that network object while
    omitting its ``docker0`` interface; ordinary Compose networks still work.
    Running the same probe in Docker's ``none`` namespace avoids that false
    negative, after which Harbor's normal nftables egress sidecar is unchanged.
    """

    @staticmethod
    def _egress_control_kernel_support() -> bool:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "container",
                    "run",
                    "--network",
                    "none",
                    "--rm",
                    DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE,
                    "sh",
                    "-c",
                    DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return False
        return result.returncode == 0

    async def _ensure_egress_control_sidecar_image_built(self) -> None:
        """Build Harbor's sidecar even when the optional buildx plugin is absent."""
        buildx = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if buildx.returncode == 0:
            await super()._ensure_egress_control_sidecar_image_built()
            return

        context = self._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH
        dockerfile = self._egress_control_sidecar_dockerfile_path()
        platform = await default_docker_platform()
        digest = docker_build_context_hash(
            context=context,
            dockerfile_path=dockerfile,
            build_args={},
            platform=platform,
        )
        image = f"{self._EGRESS_CONTROL_SIDECAR_DOCKER_NAME}--{digest}"
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await inspect.wait() != 0:
            # Harbor's pinned Dockerfile uses BuildKit-only COPY --chmod. Make
            # an equivalent temporary context for Docker's legacy builder.
            with tempfile.TemporaryDirectory() as temporary:
                portable_context = Path(temporary) / "context"
                shutil.copytree(context, portable_context)
                portable_dockerfile = portable_context / "Dockerfile"
                portable = portable_dockerfile.read_text()
                portable = portable.replace("COPY --chmod=755 ", "COPY ")
                portable += (
                    "\nRUN chmod 755 /opt/egress-sidecar/entrypoint.sh "
                    "/usr/local/bin/network-policy\n"
                )
                portable_dockerfile.write_text(portable)
                build = await asyncio.create_subprocess_exec(
                    "docker",
                    "build",
                    "--network=none",
                    f"--file={portable_dockerfile}",
                    f"--platform={platform}",
                    f"--tag={image}",
                    str(portable_context),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await build.communicate()
                if build.returncode != 0:
                    raise RuntimeError(
                        "Failed to build Harbor egress sidecar with Docker's "
                        f"standard builder: {stdout.decode(errors='replace')}"
                    )
        self._env_vars.egress_control_sidecar_image_name = image
