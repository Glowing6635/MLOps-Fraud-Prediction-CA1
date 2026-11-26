"""Load the PBI #25 engineer roster and candidate manifest from `config/retrain_roster.toml`.

Factories are looked up by name in `models.py` only, so the file cannot import
arbitrary code. `build_roster()` returns what `EvaluationInputs` needs besides the
frozen validation frame and champion, which the caller supplies.
"""

import tomllib
from pathlib import Path

from . import models
from .retrain import ModelSpec, validate_manifest

DEFAULT_PATH = Path(__file__).parents[2] / "config" / "retrain_roster.toml"


def load_roster(
    path: str | Path = DEFAULT_PATH,
) -> tuple[list[str], list[ModelSpec]]:
    """Return `(engineers, models)`, checking that every engineer has a model."""
    with Path(path).open("rb") as source:
        settings = tomllib.load(source)
    if set(settings) != {"engineers", "models"}:
        raise ValueError("Expected only `engineers` and `models` in the roster file.")
    specs = []
    for entry in settings["models"]:
        if set(entry) != {"name", "owner", "factory", "params"}:
            raise ValueError(f"Model entry needs name, owner, factory, params: {entry}")
        factory = entry["factory"]
        if not factory.endswith("_model") or not callable(
            build := getattr(models, factory, None)
        ):
            raise ValueError(f"Unknown model factory: {factory}")
        specs.append(ModelSpec(entry["name"], entry["owner"], build, entry["params"]))
    engineers = list(settings["engineers"])
    validate_manifest(specs, engineers)
    return engineers, specs
