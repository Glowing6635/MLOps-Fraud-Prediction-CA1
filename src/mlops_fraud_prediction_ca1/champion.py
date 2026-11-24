"""Pick the top 2 eligible runs, register them and move the champion alias (PBI #15).

Eligible runs are validation runs from the #13 harness that are tagged
`champion_eligible` = "True" (never the exploratory, with-customer-columns runs of
#14) and were evaluated at the same K. They are ranked like `top_k.rank_models`:
most fraud caught at K, then higher average precision.

Registry layout: one registered model; the best version carries the `champion`
alias, the second carries `challenger`. Every version is tagged with its source run,
the artifact chosen and its metrics. Each alias move records the previous champion
version on the new one (`previous_champion_version`), so rollback is a move back.
"""

from dataclasses import dataclass

CHAMPION = "champion"
CHALLENGER = "challenger"


@dataclass(frozen=True)
class Candidate:
    run_id: str
    name: str
    fraud_caught_at_k: float
    average_precision: float
    k: str
    artifact_uri: str


def eligible_candidates(client, experiment_ids, k: int) -> list[Candidate]:
    """Eligible runs at `k`, best first. Raises `ValueError` if fewer than 2 exist."""
    runs = client.search_runs(
        experiment_ids,
        filter_string="tags.champion_eligible = 'True'",
        max_results=1000,
    )
    found = []
    for run in runs:
        params, metrics = run.data.params, run.data.metrics
        if params.get("exploratory") == "True" or params.get("k") != str(k):
            continue
        if "fraud_caught_at_k" not in metrics:
            continue
        found.append(
            Candidate(
                run.info.run_id,
                _name(client, run),
                metrics["fraud_caught_at_k"],
                metrics.get("average_precision", 0.0),
                params["k"],
                f"runs:/{run.info.run_id}/model",
            )
        )
    found.sort(key=lambda c: (-c.fraud_caught_at_k, -c.average_precision, c.run_id))
    if len(found) < 2:
        raise ValueError(f"Need at least 2 eligible runs at K={k}; found {len(found)}.")
    return found


def _name(client, run) -> str:
    """The parent run's name (the model) when there is one, else the run's own name."""
    parent = run.data.tags.get("mlflow.parentRunId")
    source = client.get_run(parent) if parent else run
    return source.data.tags.get("mlflow.runName", "")


def comparison_table(candidates: list[Candidate], chosen: int = 2) -> str:
    """Markdown table of the ranked candidates with the reason for the order."""
    lines = [
        "| Rank | Run | Fraud caught at K | Average precision | Registered |",
        "|---|---|---|---|---|",
    ]
    for rank, c in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | {c.name} ({c.run_id}) | {c.fraud_caught_at_k:g} |"
            f" {c.average_precision:.4f} | {'yes' if rank <= chosen else 'no'} |"
        )
    best = candidates[0]
    second = candidates[1]
    tie = best.fraud_caught_at_k == second.fraud_caught_at_k
    reason = (
        "same fraud caught at K, so the higher average precision decides"
        if tie
        else "more fraud caught at K"
    )
    lines.append(f"\nChampion: {best.name} ({best.run_id}): {reason}.")
    return "\n".join(lines)


def register_top_two(client, model_name: str, candidates: list[Candidate]) -> dict:
    """Register the top 2 as versions of `model_name`; alias champion and challenger.

    Returns {"champion": version, "challenger": version}. Registering the same runs
    again reuses their existing versions instead of creating duplicates.
    """
    if not client.search_registered_models(f"name='{model_name}'"):
        client.create_registered_model(model_name)
    existing = {
        v.run_id: v.version
        for v in client.search_model_versions(f"name='{model_name}'")
    }
    versions = {}
    for role, cand in zip((CHAMPION, CHALLENGER), candidates[:2], strict=True):
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
            "average_precision": f"{cand.average_precision:.6f}",
        }.items():
            client.set_model_version_tag(model_name, version, key, value)
        versions[role] = version
    previous = _alias_version(client, model_name, CHAMPION)
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
    becomes `challenger`, so the two registered models swap roles.
    """
    previous = _alias_version(client, model_name, CHAMPION)
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


def _alias_version(client, model_name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except Exception:  # noqa: BLE001 - MLflow raises when the alias is unset
        return None
