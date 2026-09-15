"""
Minimal dataset metadata classes for BEHAVIOR-1K evaluation.

Provides BehaviorLerobotDatasetMetadata as a lightweight alternative
to loading the full HuggingFace demo dataset at serve time.
Task prompts are sourced from the challenge demo metadata.
"""

# Natural-language task prompts used for language conditioning.
# Source: behavior-1k/2025-challenge-demos tasks.parquet
_TASK_PROMPTS = {
    "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
    "picking_up_trash": "Pick up the trash items from the floor and place them inside the trash can.",
    "slicing_vegetables": "Slice the vegetables on the cutting board using the knife.",
    "chopping_apples": "Chop the apples into pieces using the knife on the cutting board.",
    "wiping_countertop": "Wipe the countertop clean using the sponge or cloth.",
    "washing_dishes": "Wash the dirty dishes in the sink using soap and water.",
    "cleaning_bathroom": "Clean the bathroom surfaces using the cleaning supplies.",
    "folding_clothes": "Fold the clothes and place them in the drawer or shelf.",
    "making_coffee": "Make a cup of coffee using the coffee machine.",
    "setting_table": "Set the table with plates, cups, and utensils for a meal.",
}


class BehaviorLerobotDatasetMetadata:
    """
    Lightweight metadata wrapper for BEHAVIOR-1K evaluation tasks.

    Returns task prompt strings without requiring the full demo dataset
    to be locally available. Falls back to a generated prompt for tasks
    not in the known-prompts table.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        tasks: list[str] | str | None = None,
        modalities: list | None = None,
        cameras: list | None = None,
    ) -> None:
        if isinstance(tasks, str):
            self._tasks = [tasks]
        elif isinstance(tasks, list):
            self._tasks = tasks
        else:
            self._tasks = ["turning_on_radio"]

    @property
    def tasks(self) -> dict[str, str]:
        """Returns {task_name: prompt_string} for each requested task."""
        return {
            task: _TASK_PROMPTS.get(task, task.replace("_", " ").capitalize() + ".")
            for task in self._tasks
        }
