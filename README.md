# agent-runner

`agent-runner` is a small execution layer for long-running model-development
tasks written in the native [Harbor task format](https://www.harborframework.com/docs/tasks).
It does not define another task schema and does not copy evaluator logic into the
runner.

A compatible model-development task declares `/workspace/submission` as a Harbor
artifact and uses `verifier.environment_mode = "separate"`. Everything inside the
submission directory is task-defined; for FoldBench, its `contract.md` requires
`infer.sh INPUT_DIR OUTPUT_DIR`.

The runner adds the lifecycle that model R&D needs:

1. start one detached Harbor agent session with no runner-imposed deadline;
2. expose `/workspace/submission` on the host while the agent is still running;
3. publish stable, immutable snapshots without asking the agent to manage a lock;
4. replay one selected snapshot in a new Harbor trial and the task's separate
   verifier environment.

The current implementation pins Harbor `0.22.0`. Codex, OpenCode,
Mini-SWE-Agent, Gemini CLI, Claude Code, and custom Harbor agents are selected by
name; their installation and execution remain Harbor's responsibility.

## Install

Python 3.12 and Docker Compose are required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Start a FoldBench development session

The large dataset is external to both repositories. A task-owned materializer can
populate it from a pinned Hugging Face dataset revision:

```bash
agent-runner materialize \
  --task ../ai4sci-tasks/foldbench \
  --data-dir /mnt/large-disk/foldbench-data \
  --env FOLDBENCH_HF_REPO=owner/dataset \
  --env FOLDBENCH_HF_REVISION=0123456789abcdef0123456789abcdef01234567
```

The task currently contains HF placeholders, so that command intentionally fails
until the repository and immutable commit are supplied.

`--data-dir` is passed through the environment variable named by
`--data-env-var` (`FOLDBENCH_DATA_DIR` by default). Other Harbor tasks can select
their own variable name; additional non-secret task variables use repeatable
`--env KEY=VALUE` on both `start` and `evaluate`.

Start Codex after securing the operator-owned authentication file:

```bash
chmod 600 /home/me/auth2.json

agent-runner start \
  --task ../ai4sci-tasks/foldbench \
  --run-dir /mnt/large-disk/agent-runs/foldbench-codex-001 \
  --data-dir /mnt/large-disk/foldbench-data \
  --agent codex \
  --model openai/gpt-5.6-sol \
  --auth-file /home/me/auth2.json \
  --network allowlist \
  --allow-host api.openai.com \
  --gpus task
```

`--network no-network` is the default. A remote CLI needs `allowlist` plus all
provider and first-run installation hosts it actually uses. `--network public` is
an explicit, warned escape hatch. The verifier is forced offline in runner-staged
tasks. Authentication contents and `--env` values are inherited by the detached
worker but are not stored in `run.json` or mounted into the model workspace.

For `no-network`, the staged Compose files use `network_mode: none`; this does not
depend on Docker's default bridge. `allowlist` uses Harbor's nftables sidecar and
therefore requires the host's ordinary Docker bridge/network stack to work.

Harbor's local Docker provider does not translate task GPU metadata into Compose
device requests. The runner therefore stages a private task copy and adds the
requested NVIDIA device reservation to both development and verifier Compose
files. The source task remains unchanged.

## Observe, snapshot, stop

```bash
agent-runner status /mnt/large-disk/agent-runs/foldbench-codex-001
agent-runner snapshot /mnt/large-disk/agent-runs/foldbench-codex-001
agent-runner stop /mnt/large-disk/agent-runs/foldbench-codex-001
```

The live mutable directory is `RUN/live/submission`. A background watcher hashes
it twice, copies it, verifies the source and copy again, and only then atomically
updates `RUN/snapshots/latest.json`. An escaping symlink or a concurrent write is
rejected/retried. Each `RUN/snapshots/ID/submission` is immutable by convention
and content-addressed in its adjacent manifest.

## Independent evaluation

```bash
agent-runner evaluate \
  /mnt/large-disk/agent-runs/foldbench-codex-001 \
  --snapshot latest
```

Evaluation verifies the snapshot checksum, mounts the snapshot read-only in a new
Harbor trial, copies it to the declared artifact path, and lets the task's own
separate verifier run. It verifies the checksum again afterwards. Results live in
`RUN/evaluations/SNAPSHOT_ID/evaluation.json` and the underlying Harbor trial
directory. The evaluator never receives the mutable development workspace.

## Run layout

```text
RUN/
  run.json                 non-secret resolved runner inputs
  state.json               current session state
  runner.log               detached worker log
  runtime-task/            staged Harbor task; source task is untouched
  live/submission/         mutable delivery directory visible to the agent
  snapshots/latest.json    atomic pointer
  snapshots/ID/
    manifest.json
    submission/            immutable evaluator input
  harbor/                  development Harbor trial
  evaluations/             independent verifier trials
```

The runner does not choose a single scalar across multi-metric tasks. For
FoldBench, `evaluation.json` preserves the nine task-defined track rewards; an
experiment controller can decide which evaluated snapshot is "best" without
changing the artifact contract.

## Validation levels

`pytest` runs fast unit tests. The opt-in Docker smoke uses the real FoldBench
Harbor task, real development and separate-verifier images, one native fixture,
and a deliberately trivial replay submission. It proves runtime plumbing only;
the copied native is not a trained model or a benchmark result.

```bash
RUN_FOLDBENCH_SMOKE=1 \
FOLDBENCH_TASK_DIR=../ai4sci-tasks/foldbench \
FOLDBENCH_SMOKE_SUITE_DIR=/path/to/foldbench-full/suite \
FOLDBENCH_SMOKE_GROUND_TRUTH_DIR=/path/to/ground_truths \
FOLDBENCH_SMOKE_GPUS=8 \
pytest -m foldbench_smoke -q
```
