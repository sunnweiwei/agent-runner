from __future__ import annotations

from collections.abc import Sequence
import subprocess

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkPolicy


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
