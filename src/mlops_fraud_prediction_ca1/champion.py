"""Pick the top 2 eligible runs, register them and move the champion alias (PBI #15).

Eligible runs are validation runs from the #13 harness that are tagged
`champion_eligible` = "True" (never the exploratory, with-customer-columns runs of
#14) and were evaluated at the same K on the same validation data. They are ranked like
`top_k.rank_models`:
most fraud caught at K, then higher average precision.

Registry layout: one registered model; the best version carries the `champion`
alias, the second carries `challenger`. Every version is tagged with its source run,
the artifact chosen and its metrics. Each alias move records the previous champion
version on the new one (`previous_champion_version`), so rollback is a move back.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from math import isfinite
from re import fullmatch

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"
CHALLENGER = "challenger"
TOP_TWO = 2
MAX_MODEL_SIZE_BYTES = 50 * 1024 * 1024
DATASET_HASHES = tuple(
    f"dataset_sha256_{table}" for table in ("customers", "orders", "labels")
)


@dataclass(frozen=True)
class Candidate:
    run_id: str
    name: str
    fraud_caught_at_k: float
    average_precision: float | None
    k: str
    artifact_uri: str


def eligible_candidates(client, experiment_ids, k: int) -> list[Candidate]:
    """Rank comparable runs at `k` and require at least two eligible candidates."""
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer.")
    found = []
    window = None
    for run in _pages(
        client.search_runs,
        experiment_ids,
        filter_string="tags.champion_eligible = 'True'",
    ):
        params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
        if (
            run.info.status != "FINISHED"
            or params.get("exploratory") != "False"
            or tags.get("git_dirty") != "False"
        ):
            continue
        run_id = run.info.run_id
        if "k" not in params:
            raise ValueError(f"Run {run_id} has no recorded K.")
        if params["k"] != str(k):
            continue
        model_size = metrics.get("model_size_bytes")
        if model_size is not None and model_size > MAX_MODEL_SIZE_BYTES:
            continue
        provenance = _validation_provenance(run_id, params, metrics)
        if window is not None and provenance != window:
            raise ValueError(
                f"Run {run_id} has different validation provenance at K={k}."
            )
        window = provenance
        found.append(
            Candidate(
                run_id,
                _name(client, run),
                metrics["fraud_caught_at_k"],
                metrics.get("average_precision"),
                params["k"],
                f"runs:/{run_id}/model",
            )
        )
    found.sort(
        key=lambda c: (-c.fraud_caught_at_k, -(c.average_precision or 0), c.run_id)
    )
    if len(found) < TOP_TWO:
        raise ValueError(f"Need at least 2 eligible runs at K={k}; found {len(found)}.")
    return found


def _validation_provenance(run_id, params, metrics) -> tuple:
    """Check the harness inputs and the #5 order and fraud totals."""
    try:
        hashes = tuple(params[key] for key in DATASET_HASHES)
        start = date.fromisoformat(params["validation_start"])
        end = date.fromisoformat(params["test_start"])
        policy = params["cleaning_policy_version"]
        orders = metrics["validation_orders"]
        fraud = metrics["total_fraud_at_k"]
        caught = metrics["fraud_caught_at_k"]
        ap = metrics.get("average_precision")
        if (
            all(fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes)
            and start < end
            and policy.strip()
            and all(
                isfinite(count) and count == int(count)
                for count in (orders, fraud, caught)
            )
            and 0 <= caught <= fraud <= orders
            and orders > 0
            and (fraud == 0 if ap is None else 0 <= ap <= 1)
        ):
            return hashes, start, end, policy, orders, fraud
    except (KeyError, ValueError):
        pass
    raise ValueError(
        f"Run {run_id} has missing or invalid validation provenance or metrics."
    )


def _pages(search, *args, **kwargs):
    token = None
    while True:
        page = search(*args, **kwargs, max_results=1000, page_token=token)
        yield from page
        token = page.token
        if not token:
            return


def _name(client, run) -> str:
    """The parent run's name (the model) when there is one, else the run's own name."""
    parent = run.data.tags.get("mlflow.parentRunId")
    source = client.get_run(parent) if parent else run
    return source.data.tags.get("mlflow.runName", "")


def comparison_table(candidates: list[Candidate]) -> str:
    """Markdown table of the ranked candidates with the reason for the order."""
    lines = [
        "| Rank | Run | Fraud caught at K | Average precision | Top two |",
        "|---|---|---|---|---|",
    ]
    for rank, c in enumerate(candidates, 1):
        ap = "N/A" if c.average_precision is None else f"{c.average_precision:.4f}"
        lines.append(
            f"| {rank} | {c.name} ({c.run_id}) | {c.fraud_caught_at_k:g} |"
            f" {ap} | {'yes' if rank <= TOP_TWO else 'no'} |"
        )
    best = candidates[0]
    second = candidates[1]
    if best.fraud_caught_at_k != second.fraud_caught_at_k:
        reason = "more fraud caught at K"
    elif best.average_precision != second.average_precision:
        reason = "same fraud caught at K, so the higher average precision decides"
    else:
        reason = (
            "same fraud caught at K and average precision, so run ID breaks the tie"
        )
    lines.append(f"\nChampion: {best.name} ({best.run_id}): {reason}.")
    return "\n".join(lines)


def register_top_two(client, model_name: str, candidates: list[Candidate]) -> dict:
    """Register the top 2 as versions of `model_name`; alias champion and challenger.

    Returns {"champion": version, "challenger": version}. Registering the same runs
    again reuses their existing versions instead of creating duplicates.
    """
    aliases = _registered_aliases(client, model_name)
    if aliases is None:
        client.create_registered_model(model_name)
        aliases = {}
    previous = aliases.get(CHAMPION)
    existing = {}
    for version in _pages(client.search_model_versions, f"name='{model_name}'"):
        existing.setdefault(version.run_id, version.version)
    versions = {}
    for role, cand in zip((CHAMPION, CHALLENGER), candidates[:TOP_TWO], strict=True):
        version = str(
            existing.get(cand.run_id)
            or client.create_model_version(
                model_name, cand.artifact_uri, run_id=cand.run_id
            ).version
        )
        for key, value in {
            "source_run_id": cand.run_id,
            "artifact": cand.artifact_uri,
            "k": cand.k,
            "fraud_caught_at_k": f"{cand.fraud_caught_at_k:g}",
            "average_precision": (
                "N/A"
                if cand.average_precision is None
                else f"{cand.average_precision:.6f}"
            ),
        }.items():
            client.set_model_version_tag(model_name, version, key, value)
        versions[role] = version
    for role, version in versions.items():
        if role == CHAMPION and previous and previous != version:
            client.set_model_version_tag(
                model_name, version, "previous_champion_version", previous
            )
        client.set_registered_model_alias(model_name, role, version)
    return versions


def move_champion(client, model_name: str, to_version: str, reason: str) -> str:
    """Point `champion` at `to_version`, recording the previous version; return it.

    Rollback is the same call with the recorded previous version. The old champion
    becomes `challenger`, replacing any version previously carrying that alias.
    """
    aliases = _registered_aliases(client, model_name) or {}
    previous = aliases.get(CHAMPION)
    if previous is None:
        raise ValueError(f"{model_name} has no champion to move.")
    if previous == str(to_version):
        return previous
    client.get_model_version(model_name, str(to_version))  # fails if it is unknown
    client.set_model_version_tag(
        model_name, str(to_version), "previous_champion_version", previous
    )
    client.set_model_version_tag(
        model_name, str(to_version), "champion_change_reason", reason
    )
    client.set_registered_model_alias(model_name, CHAMPION, str(to_version))
    client.set_registered_model_alias(model_name, CHALLENGER, previous)
    return previous


def _registered_aliases(client, model_name: str) -> dict[str, str] | None:
    try:
        model = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None
    return {alias: str(version) for alias, version in model.aliases.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eligible MLflow models.")
    parser.add_argument("--k", type=int, required=True, help="Daily review capacity.")
    parser.add_argument(
        "--register",
        metavar="MODEL_NAME",
        help="Register the top two under MODEL_NAME and update the aliases.",
    )
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not uri or not experiment_name:
        parser.error("Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME.")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        parser.error(f"MLflow experiment {experiment_name!r} does not exist.")

    candidates = eligible_candidates(client, [experiment.experiment_id], args.k)
    sys.stdout.write(comparison_table(candidates) + "\n")
    if args.register:
        versions = register_top_two(client, args.register, candidates)
        for alias, version in versions.items():
            sys.stdout.write(f"{args.register} {alias}: version {version}\n")
