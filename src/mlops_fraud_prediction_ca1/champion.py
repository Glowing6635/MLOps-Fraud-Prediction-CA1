"""Pick the top 2 eligible runs, register them and move the champion alias (PBI #15).

Eligible runs are validation runs from the #13 harness that are tagged
`champion_eligible` = "True" (never the exploratory, with-customer-columns runs of
#14) and have comparable validation evidence at the same K. They are ranked by
most fraud caught at K, then higher average precision and then run ID. The
registry gets the best two of them whose model is within 50 MB (D15-03).

Registry layout: one registered model; the best version carries the `champion`
alias, the second carries `challenger`. Every version is tagged with its source run,
the artifact chosen and its metrics. Each alias move records the previous champion
version on the new one (`previous_champion_version`), so rollback is a move back.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from math import isfinite
from re import fullmatch

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.artifacts import list_artifacts
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.utils.uri import append_to_uri_path

CHAMPION = "champion"
CHALLENGER = "challenger"
TOP_TWO = 2
SIZE_NOTE_BYTES = 50_000_000
DATASET_HASHES = tuple(
    f"dataset_sha256_{table}" for table in ("customers", "orders", "labels")
)


@dataclass(frozen=True)
class _ValidationProvenance:
    dataset_hashes: tuple[str, ...]
    start: str
    end: str
    cleaning_policy: str
    orders: int
    fraud: int
    rows_sha256: str | None


@dataclass(frozen=True)
class Candidate:
    run_id: str
    name: str
    fraud_caught_at_k: float
    average_precision: float | None
    k: str
    artifact_uri: str
    validation: _ValidationProvenance
    fraud_catchable_at_k: int | None
    saved_model_size_bytes: int | None


def eligible_candidates(client, experiment_ids, k: int) -> list[Candidate]:
    """Rank comparable runs at `k` and require at least two eligible candidates."""
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer.")
    found = []
    window = None
    catchable_at_k = None
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
        recorded_k = params.get("k", "")
        if not fullmatch(r"[1-9][0-9]*", recorded_k):
            raise ValueError(f"Run {run_id} has missing or invalid recorded K.")
        if recorded_k != str(k):
            continue
        provenance = _validation_provenance(run_id, params, metrics)
        if window is not None and provenance != window:
            raise ValueError(
                f"Run {run_id} has different validation provenance at K={k}. "
                "Compare matching inputs, windows, counts and row fingerprints "
                "in one experiment."
            )
        window = provenance
        catchable = metrics.get("fraud_catchable_at_k")
        if catchable is not None:
            if catchable_at_k is not None and catchable != catchable_at_k:
                raise ValueError(
                    f"Run {run_id} has different fraud catchable at K={k}."
                )
            catchable_at_k = int(catchable)
        size = metrics.get("model_size_bytes")
        found.append(
            Candidate(
                run_id=run_id,
                name=_name(client, run),
                fraud_caught_at_k=metrics["fraud_caught_at_k"],
                average_precision=metrics.get("average_precision"),
                k=recorded_k,
                artifact_uri=f"runs:/{run_id}/model",
                validation=provenance,
                fraud_catchable_at_k=None if catchable is None else int(catchable),
                saved_model_size_bytes=int(size) if _whole_nonnegative(size) else None,
            )
        )
    found.sort(
        key=lambda c: (-c.fraud_caught_at_k, -(c.average_precision or 0), c.run_id)
    )
    if len(found) < TOP_TWO:
        raise ValueError(f"Need at least 2 eligible runs at K={k}; found {len(found)}.")
    return found


def _whole_nonnegative(value):
    return value is not None and isfinite(value) and value >= 0 and value == int(value)


def _validation_provenance(run_id, params, metrics) -> _ValidationProvenance:
    """Check the harness inputs and the #5 order and fraud totals."""
    try:
        hashes = tuple(params[key] for key in DATASET_HASHES)
        start = date.fromisoformat(params["validation_start"])
        end = date.fromisoformat(params["test_start"])
        policy = params["cleaning_policy_version"]
        rows = params.get("validation_rows_sha256")
        orders = metrics["validation_orders"]
        fraud = metrics["total_fraud_at_k"]
        caught = metrics["fraud_caught_at_k"]
        ap = metrics.get("average_precision")
        catchable = metrics.get("fraud_catchable_at_k")
        if (
            all(fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes)
            and start < end
            and policy.strip()
            and (rows is None or fullmatch(r"[0-9a-f]{64}", rows))
            and all(_whole_nonnegative(count) for count in (orders, fraud, caught))
            and caught <= fraud <= orders
            and orders > 0
            and (ap is None if fraud == 0 else ap is not None and 0 <= ap <= 1)
            and (
                catchable is None
                or (_whole_nonnegative(catchable) and caught <= catchable <= fraud)
            )
        ):
            return _ValidationProvenance(
                hashes,
                start.isoformat(),
                end.isoformat(),
                policy,
                int(orders),
                int(fraud),
                rows,
            )
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


def comparison_table(
    candidates: list[Candidate], *, elected: Candidate | None = None
) -> str:
    """Show the validation ranking and the elected registry champion when known."""
    lines = [
        (
            "| Rank | Run | Fraud caught at K | Fraud catchable at K | Average precision "
            "| Saved model size (harness) | Top two by rank |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for rank, c in enumerate(candidates, 1):
        ap = "N/A" if c.average_precision is None else f"{c.average_precision:.4f}"
        catchable = (
            "unknown" if c.fraud_catchable_at_k is None else c.fraud_catchable_at_k
        )
        lines.append(
            f"| {rank} | {c.name} ({c.run_id}) | {c.fraud_caught_at_k:g} |"
            f" {catchable} | {ap} | {_size_note(c.saved_model_size_bytes)} |"
            f" {'yes' if rank <= TOP_TWO else 'no'} |"
        )
    if candidates[0].validation.rows_sha256 is None:
        lines.append(
            "\nValidation provenance: matching harness inputs, window and counts. "
            "Exact scored-row identity is unverified (legacy runs)."
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
    lines.append(f"\nRank 1: {best.name} ({best.run_id}): {reason}.")
    if elected is not None:
        rank = candidates.index(elected) + 1
        lines.append(f"Champion: {elected.name} ({elected.run_id}) (rank {rank}).")
    return "\n".join(lines)


def _size_note(size: int | None) -> str:
    if size is None:
        return "unknown"
    suffix = " (>50 MB)" if size > SIZE_NOTE_BYTES else ""
    return f"{size:,} bytes{suffix}"


@dataclass(frozen=True)
class _ModelSource:
    location: str
    logged_id: str | None = None


def _artifact_source(client, candidate: Candidate) -> _ModelSource:
    run_id = candidate.run_id
    if candidate.artifact_uri != f"runs:/{run_id}/model":
        raise ValueError(f"Unexpected elected artifact for run {run_id}.")
    info = client.get_run(run_id).info
    logged = list(
        _pages(
            client.search_logged_models,
            [info.experiment_id],
            filter_string=f"name = 'model' AND source_run_id = '{run_id}'",
        )
    )
    legacy_source = append_to_uri_path(info.artifact_uri, "model")
    # Listing a runs:/ URI can mix legacy files with logged-model artifacts.
    legacy = _has_manifest(
        list_artifacts(artifact_uri=legacy_source, tracking_uri=client.tracking_uri),
        "model/MLmodel",
    )
    if len(logged) + legacy != 1:
        raise ValueError(f"Run {run_id} must identify exactly one model artifact.")
    if legacy:
        return _ModelSource(legacy_source)

    model = logged[0]
    if model.status.name != "READY" or not _has_manifest(
        client.list_logged_model_artifacts(model.model_id), "MLmodel"
    ):
        raise ValueError(f"Run {run_id} has an incomplete logged model.")
    return _ModelSource(model.artifact_location, model.model_id)


def _has_manifest(artifacts, path: str) -> bool:
    return any(artifact.path == path and not artifact.is_dir for artifact in artifacts)


def _artifact_size(client, source: _ModelSource) -> int | None:
    """Count bytes from the exact source's inventory, without downloading it."""
    try:
        repository = get_artifact_repository(
            source.location, tracking_uri=client.tracking_uri
        )
        size = 0
        directories = [None]
        while directories:
            entries = repository.list_artifacts(directories.pop())
            if not entries:
                return None
            for entry in entries:
                if entry.is_dir:
                    directories.append(entry.path)
                elif not _whole_nonnegative(entry.file_size):
                    return None
                else:
                    size += entry.file_size
        return size
    except Exception:  # noqa: BLE001
        # Optional inventory reads can raise backend-specific SDK errors.
        return None


def _registration_picks(
    client, candidates: list[Candidate]
) -> tuple[list[Candidate], list[_ModelSource], list[int | None]]:
    """Take the best two ranked candidates whose model fits within 50 MB.

    Every candidate stays in the ranking. A candidate is passed over
    when its exact artifact size, or failing that the harness's saved-copy size,
    is known and above 50 MB. An unmeasured size shows as unknown and does not
    block registration.
    """
    picked, sources, sizes = [], [], []
    for candidate in candidates:
        saved = candidate.saved_model_size_bytes
        try:
            source = _artifact_source(client, candidate)
        except ValueError:
            if saved is not None and saved > SIZE_NOTE_BYTES:
                continue  # oversized and not registrable anyway
            raise
        exact = _artifact_size(client, source)
        size = saved if exact is None else exact
        if size is not None and size > SIZE_NOTE_BYTES:
            continue
        picked.append(candidate)
        sources.append(source)
        sizes.append(exact)
        if len(picked) == TOP_TWO:
            return picked, sources, sizes
    raise ValueError("Fewer than two ranked candidates are within 50 MB.")


def comparison_id(client, candidates: list[Candidate]) -> str:
    """Fingerprint the comparison and the two resolved sources for analyst approval."""
    picked, sources, _ = _registration_picks(client, candidates)
    return _comparison_id(client, candidates, picked, sources)


def _comparison_id(client, candidates, picked, sources) -> str:
    ranking = [asdict(candidate) for candidate in candidates]
    for entry in ranking:
        # Size is informational. It may be measured or corrected after approval.
        del entry["saved_model_size_bytes"]
    payload = {
        "schema": 2,
        "tracking_uri": client.tracking_uri,
        "ranking": ranking,
        "registered": [candidate.run_id for candidate in picked],
        "sources": [asdict(source) for source in sources],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _matches_artifact(client, model_name, version, candidate, source) -> bool:
    if version.run_id != candidate.run_id or version.status != "READY":
        return False
    if source.logged_id is not None:
        # A runs:/ source can fall back to a legacy location when loaded
        # through a registry alias. Require the logged model's pinned location.
        references = {source.location, f"models:/{source.logged_id}"}
        downloads = {source.location}
    else:
        references = downloads = {source.location, candidate.artifact_uri}
    return (
        version.source in references
        and client.get_model_version_download_uri(model_name, version.version)
        in downloads
    )


def register_top_two(
    client, model_name: str, candidates: list[Candidate], *, approved_comparison: str
) -> dict[str, str]:
    """Register the approved top two artifacts and assign their aliases."""
    picked, sources, _ = _registration_picks(client, candidates)
    return _register_approved(
        client, model_name, candidates, picked, sources, approved_comparison
    )


def _register_approved(
    client, model_name, candidates, picked, sources, approved_comparison
):
    if approved_comparison != _comparison_id(client, candidates, picked, sources):
        raise ValueError("Comparison changed. Preview it again and obtain approval.")
    aliases = _registered_aliases(client, model_name)
    if aliases is None:
        client.create_registered_model(model_name)
        aliases = {}
    existing = sorted(
        _pages(client.search_model_versions, f"name='{model_name}'"),
        key=lambda version: int(version.version),
    )
    versions = {}
    for role, cand, source in zip((CHAMPION, CHALLENGER), picked, sources, strict=True):
        matches = [
            v
            for v in existing
            if _matches_artifact(client, model_name, v, cand, source)
        ]
        version = next(
            (v for v in matches if str(v.version) == aliases.get(role)),
            matches[0] if matches else None,
        )
        if version is None:
            version = client.create_model_version(
                model_name,
                source.location,
                run_id=cand.run_id,
                model_id=source.logged_id,
            )
            if not _matches_artifact(client, model_name, version, cand, source):
                raise ValueError(
                    f"Registry version {version.version} is not ready at the expected source."
                )
        version_number = str(version.version)
        for key, value in {
            "source_run_id": cand.run_id,
            "artifact": cand.artifact_uri,
            "resolved_artifact": source.location,
            "k": cand.k,
            "fraud_caught_at_k": f"{cand.fraud_caught_at_k:g}",
            "average_precision": (
                "N/A"
                if cand.average_precision is None
                else f"{cand.average_precision:.6f}"
            ),
        }.items():
            client.set_model_version_tag(model_name, version_number, key, value)
        versions[role] = version_number
    _set_aliases(client, model_name, aliases, versions)
    return versions


def move_champion(client, model_name: str, to_version: str, reason: str) -> str:
    """Move the champion alias and return the previous version.

    Rollback is the same call with the recorded previous version. The old champion
    becomes `challenger`, replacing any version previously carrying that alias.
    """
    aliases = _registered_aliases(client, model_name) or {}
    previous = aliases.get(CHAMPION)
    if previous is None:
        raise ValueError(f"{model_name} has no champion to move.")
    destination = str(to_version)
    if previous == destination and aliases.get(CHALLENGER) != destination:
        return previous

    target = client.get_model_version(model_name, destination)
    if previous == destination:
        # Older code wrote champion first. Its tags identify an interrupted
        # move when the operator retries the same destination and reason.
        previous = target.tags.get("previous_champion_version")
        if (
            not previous
            or previous == destination
            or target.tags.get("champion_change_reason") != reason
        ):
            raise ValueError(f"{model_name} has an ambiguous interrupted alias move.")
        client.get_model_version(model_name, previous)
    else:
        client.set_model_version_tag(
            model_name, destination, "champion_change_reason", reason
        )

    _set_aliases(
        client, model_name, aliases, {CHAMPION: destination, CHALLENGER: previous}
    )
    return previous


def _set_aliases(client, model_name, before, desired):
    previous = before.get(CHAMPION)
    destination = desired[CHAMPION]
    if previous and previous != destination:
        client.set_model_version_tag(
            model_name, destination, "previous_champion_version", previous
        )

    # Champion is the last write. Until it moves, a retry can still read
    # the previous champion even if the challenger write took effect.
    challenger_changed = before.get(CHALLENGER) != desired[CHALLENGER]
    try:
        if challenger_changed:
            client.set_registered_model_alias(
                model_name, CHALLENGER, desired[CHALLENGER]
            )
        if previous != destination:
            client.set_registered_model_alias(model_name, CHAMPION, destination)
    except Exception as error:
        if challenger_changed and previous != destination:
            _restore_challenger(client, model_name, before, desired[CHALLENGER], error)
        raise


def _restore_challenger(client, model_name, before, attempted, error):
    try:
        current = _registered_aliases(client, model_name)
        if (
            current is None
            or current.get(CHAMPION) != before.get(CHAMPION)
            or current.get(CHALLENGER) != attempted
        ):
            return
        if CHALLENGER in before:
            client.set_registered_model_alias(
                model_name, CHALLENGER, before[CHALLENGER]
            )
        else:
            client.delete_registered_model_alias(model_name, CHALLENGER)
    except Exception as restoration_error:  # noqa: BLE001 - keep the write error
        error.add_note(f"Could not restore the challenger alias: {restoration_error}")


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
        help="Register the best two ranked models within 50 MB under MODEL_NAME and update the aliases.",
    )
    parser.add_argument(
        "--approve-comparison",
        metavar="ID",
        help="Comparison ID from the analyst-approved preview. Required to register.",
    )
    args = parser.parse_args()
    if (args.register is None) != (args.approve_comparison is None):
        parser.error("--register and --approve-comparison must be supplied together.")

    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not uri or not experiment_name:
        parser.error("Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME.")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        parser.error(f"MLflow experiment {experiment_name!r} does not exist.")

    try:
        candidates = eligible_candidates(client, [experiment.experiment_id], args.k)
        picked, sources, sizes = _registration_picks(client, candidates)
        table = comparison_table(candidates, elected=picked[0])
        if args.register is None:
            approval_id = _comparison_id(client, candidates, picked, sources)
            sys.stdout.write(table + "\n\n")
            for role, candidate, source, exact in zip(
                (CHAMPION, CHALLENGER), picked, sources, sizes, strict=True
            ):
                rank = candidates.index(candidate) + 1
                uri = (
                    f"models:/{source.logged_id}"
                    if source.logged_id
                    else candidate.artifact_uri
                )
                size = _size_note(exact)
                sys.stdout.write(
                    f"{role} (rank {rank}) artifact: {uri} (artifact size: {size})\n"
                )
            sys.stdout.write(f"Comparison ID: {approval_id}\n")
            return
        versions = _register_approved(
            client, args.register, candidates, picked, sources, args.approve_comparison
        )
    except (ValueError, MlflowException) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    sys.stdout.write(table + "\n")
    for alias, version in versions.items():
        sys.stdout.write(f"{args.register} {alias}: version {version}\n")
