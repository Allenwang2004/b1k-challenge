"""Launch the RLC (1st place) policy server against our 2026 OmniGibson fork.

Their repo (github.com/IliaLarchenko/behavior-1k-solution) targets the **2025**
BEHAVIOR API, where the eval package was ``omnigibson.learning``. Upstream
renamed it to ``omnigibson.eval`` for the 2026 edition, so three of their
serving modules fail to import against our fork:

    scripts/serve_b1k.py              omnigibson.learning.utils.network_utils
                                      omnigibson.learning.datas
    src/b1k/policies/b1k_policy.py    omnigibson.learning.utils.eval_utils
    src/b1k/shared/eval_b1k_wrapper.py            "

This launcher installs an alias so ``omnigibson.learning`` resolves to
``omnigibson.eval``, then delegates to their ``serve_b1k.main`` unchanged.
Editing their source would work too, but their repo is the reference
implementation of a published result -- keeping it byte-identical means a
future ``git pull`` cannot silently drop a patch, and nobody has to wonder
whether a behavioural difference came from us. The shim is the whole delta.

``BehaviorLerobotDatasetMetadata`` is imported by their serve script but never
used on the PI_BEHAVIOR path (their model takes a task embedding, not a text
prompt), so it is stubbed rather than ported.

## What this server does that a plain openpi server does not

Their wrapper is stateful and the state is load-bearing -- it is not a thin
shell around ``policy.infer``:

* **System 2 stage predictor.** The VLM emits ``subtask_logits`` for the
  current stage (5-15 stages per task, see ``TASK_NUM_STAGES``); the wrapper
  votes over a 3-prediction history before advancing, and feeds the resulting
  stage back in as ``subtask_state``. Stage is model input, not just output.
* **Rolling soft inpainting.** Predicts 30 actions, executes 26, keeps 4 to
  condition the next prediction.
* **Cubic action compression.** 26 actions executed in 20 steps (1.3x speedup),
  automatically disabled when the gripper command is changing.
* **Correction rules**, including "open the gripper after a failed grasp".
* **Per-task checkpoints.** Four checkpoints; the wrapper switches on the
  ``task_id`` in the observation.

Because it is stateful, **one episode's state must not leak into the next**.
The server calls ``policy.reset()`` when the client reconnects, and the wrapper
also resets whenever ``task_id`` changes. Running one eval process per task (as
``eval_rollout.py`` does) satisfies both.

Verified: their task embeddings are indexed 0-49 and their 50 training tasks are
exactly ``Task ID`` 0-49 in ``B100_task_misc.csv``, the table behind our
``TASK_NAMES_TO_INDICES``. The ``task_id`` our evaluator sends
(``evaluator.py:349``) therefore lines up with both their embeddings and their
``task_checkpoint_mapping.json``. Do not add a remapping.

## Usage

All 50 tasks, switching checkpoints automatically (what the sweep wants)::

    cd <solution-repo>
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \\
    .venv/bin/python serve_ilia.py \\
        --solution-repo . \\
        --policy.config pi_behavior_b1k_fast \\
        --policy.dir <any one checkpoint, used as the initial load> \\
        --task-checkpoint-mapping ./task_checkpoint_mapping.json \\
        --port 8010

Every other flag is passed through to their ``serve_b1k.py`` untouched
(``--actions-to-execute``, ``--num-steps``, ``--apply-eval-tricks``, ...).
Defaults reproduce the submission; change them only deliberately.
"""

from __future__ import annotations

import argparse
import os
import sys
import types


def install_learning_alias() -> None:
    """Make ``omnigibson.learning`` resolve to ``omnigibson.eval``."""
    import omnigibson.eval as _eval
    import omnigibson.eval.utils as _eval_utils
    import omnigibson.eval.utils.eval_utils as _eu
    import omnigibson.eval.utils.network_utils as _nu

    sys.modules.setdefault("omnigibson.learning", _eval)
    sys.modules.setdefault("omnigibson.learning.utils", _eval_utils)
    sys.modules.setdefault("omnigibson.learning.utils.eval_utils", _eu)
    sys.modules.setdefault("omnigibson.learning.utils.network_utils", _nu)

    # Imported by their serve script but unused on the PI_BEHAVIOR path.
    datas = types.ModuleType("omnigibson.learning.datas")

    class BehaviorLerobotDatasetMetadata:  # noqa: D401 - deliberate stub
        """Unused for task-embedding models; present so the import resolves."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "BehaviorLerobotDatasetMetadata is a stub. The RLC model is task-embedding "
                "conditioned and never reads dataset prompts; if you hit this, you are on a "
                "text-prompt code path that needs a real port."
            )

    datas.BehaviorLerobotDatasetMetadata = BehaviorLerobotDatasetMetadata
    sys.modules.setdefault("omnigibson.learning.datas", datas)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--solution-repo", default=".",
                     help="checkout of IliaLarchenko/behavior-1k-solution")
    known, passthrough = pre.parse_known_args()

    repo = os.path.abspath(os.path.expanduser(known.solution_repo))
    src = os.path.join(repo, "src")
    scripts = os.path.join(repo, "scripts")
    for path in (src, scripts):
        if not os.path.isdir(path):
            raise SystemExit(f"{path} not found -- is --solution-repo pointing at the solution checkout?")
        sys.path.insert(0, path)

    install_learning_alias()

    # Import only after the alias is in place, or their imports fail.
    import tyro
    from serve_b1k import Args, main as serve_main  # noqa: N813

    sys.argv = [sys.argv[0], *passthrough]
    serve_main(tyro.cli(Args))
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, force=True)
    sys.exit(main())
