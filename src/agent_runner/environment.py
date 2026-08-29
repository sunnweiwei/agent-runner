from __future__ import annotations

from collections.abc import Sequence

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
