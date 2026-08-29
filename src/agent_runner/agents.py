from __future__ import annotations

from pathlib import Path
import shlex

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class ContinuousPublisherAgent(BaseAgent):
    """Offline smoke agent: publish a fixture, then remain alive until stopped."""

    @staticmethod
    def name() -> str:
        return "continuous-publisher"

    def __init__(
        self,
        *args,
        payload_dir: str,
        expected_gpus: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.payload_dir = Path(payload_dir).expanduser().resolve()
        self.expected_gpus = int(expected_gpus)
        if not self.payload_dir.is_dir():
            raise FileNotFoundError(f"Smoke payload does not exist: {self.payload_dir}")

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await environment.exec(command="rm -rf /workspace/submission/*")
        await environment.upload_dir(self.payload_dir, "/workspace/submission")
        await environment.exec(command="chmod +x /workspace/submission/infer.sh")
        result = await environment.exec(
            command="nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '",
        )
        actual_gpus = int((result.stdout or "0").strip() or "0")
        if actual_gpus != self.expected_gpus:
            raise RuntimeError(
                f"Expected {self.expected_gpus} visible GPU(s), found {actual_gpus}"
            )
        await environment.exec(
            command=(
                f"printf '%s\\n' {shlex.quote(str(actual_gpus))} "
                "> /workspace/submission/gpu_count.txt && "
                "printf 'ready\\n' > /workspace/submission/runner_smoke.txt && "
                "sleep infinity"
            )
        )


class ArtifactReplayAgent(BaseAgent):
    """Copy an immutable, read-only snapshot into the Harbor artifact path."""

    @staticmethod
    def name() -> str:
        return "artifact-replay"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        result = await environment.exec(
            command=(
                "set -eu; test -f /runner-input/submission/infer.sh; "
                "rm -rf /workspace/submission; "
                "mkdir -p /workspace/submission; "
                "cp -a /runner-input/submission/. /workspace/submission/"
            )
        )
        if result.return_code != 0:
            raise RuntimeError("Could not stage the immutable submission snapshot")
