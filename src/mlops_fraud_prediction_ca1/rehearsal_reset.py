"""Rebuild the rehearsal environment and serve practice batches (PBI #24).

`uv run rehearsal-reset --data-dir data` drops and recreates the rehearsal database's
tables, loads the three base CSV files with the #20 loader, and deletes the runs in the
rehearsal MLflow experiment. `uv run rehearsal-reset --presign BUCKET` uploads the
synthetic practice batches and prints presigned URLs.

The database and experiment come from the #2 settings (`DB_NAME`,
`MLFLOW_EXPERIMENT_NAME`), so pointing `.env` at the rehearsal resources selects them.
Guards run before anything is touched: both names must start with `rehearsal`, and
neither may equal the team's `TEAM_DB_NAME` or `TEAM_MLFLOW_EXPERIMENT_NAME` when set.
"""

import argparse
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

REHEARSAL_PREFIX = "rehearsal"
TABLES_DROP_ORDER = ("labels", "orders", "customers", "batches")
_BASE_FILES = ("customers_base.csv", "orders_base.csv", "labels_base.csv")


class RehearsalRefused(Exception):
    """The settings point at a database or experiment that is not a rehearsal one."""


def check_rehearsal_target(settings, environ: dict[str, str]) -> None:
    """Refuse unless both the database and the MLflow experiment are rehearsal ones."""
    checks = (
        ("database", settings.db_name, "TEAM_DB_NAME"),
        (
            "MLflow experiment",
            settings.mlflow_experiment_name,
            "TEAM_MLFLOW_EXPERIMENT_NAME",
        ),
    )
    for kind, name, team_variable in checks:
        if environ.get(team_variable) and name == environ[team_variable]:
            raise RehearsalRefused(
                f"The {kind} {name!r} is the team's ({team_variable})."
            )
        if not name.lower().startswith(REHEARSAL_PREFIX):
            raise RehearsalRefused(
                f"The {kind} {name!r} does not start with {REHEARSAL_PREFIX!r}."
            )


def reset_database(engine, drop_tables: Callable, create_tables: Callable) -> None:
    drop_tables(engine)
    create_tables(engine)


def _drop_tables(engine) -> None:
    from sqlalchemy import text

    with engine.begin() as connection:
        for table in TABLES_DROP_ORDER:  # children first: foreign keys point at parents
            connection.execute(text(f"DROP TABLE IF EXISTS `{table}`"))


def clear_experiment(client, experiment_name: str) -> int:
    """Delete every run in the experiment; return how many. A missing experiment is fine."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)
        return 0
    deleted = 0
    while True:
        runs = client.search_runs([experiment.experiment_id], max_results=1000)
        if not runs:
            return deleted
        for run in runs:
            client.delete_run(run.info.run_id)
            deleted += 1


def rebuild(
    settings,
    environ,
    data_dir: Path,
    *,
    engine,
    client,
    load_files,
    create_tables,
    drop_tables=_drop_tables,
) -> dict:
    """Guard, then rebuild the database from the base CSVs and clear the experiment."""
    check_rehearsal_target(settings, environ)
    paths = [Path(data_dir, name) for name in _BASE_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing base files: {', '.join(missing)}")
    reset_database(engine, drop_tables, create_tables)
    report = load_files(engine, *map(str, paths))
    deleted = clear_experiment(client, settings.mlflow_experiment_name)
    return {
        "loaded": report.loaded,
        "rejected": report.rejected,
        "runs_deleted": deleted,
    }


def presign_batches(
    s3, bucket: str, out_dir: Path, *, expires: int = 3600, prefix: str = "rehearsal"
) -> dict[str, list[str]]:
    """Upload each generated batch privately and return presigned GET URLs per batch.

    `s3` is a boto3 client made from the sandbox session's own credentials
    (`boto3.client("s3")`, which reads them from the environment or instance profile);
    nothing is stored in the repository, and objects are never made public.
    """
    urls = {}
    for batch_dir in sorted(p for p in Path(out_dir).iterdir() if p.is_dir()):
        urls[batch_dir.name] = []
        for file in sorted(batch_dir.glob("*.csv")):
            key = f"{prefix}/{batch_dir.name}/{file.name}"
            s3.upload_file(str(file), bucket, key)
            urls[batch_dir.name].append(
                s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expires,
                )
            )
    return urls


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--presign", metavar="BUCKET", help="serve practice batches instead"
    )
    parser.add_argument(
        "--expires", type=int, default=3600, help="link lifetime in seconds"
    )
    args = parser.parse_args(argv)

    import os

    from .config import create_db_engine, load_settings

    settings = load_settings()
    try:
        check_rehearsal_target(settings, dict(os.environ))
        if args.presign:
            import boto3

            from .rehearsal import build_batches, write_batch

            with tempfile.TemporaryDirectory() as folder:
                for batch in build_batches():
                    write_batch(batch, Path(folder))
                urls = presign_batches(
                    boto3.client("s3"), args.presign, Path(folder), expires=args.expires
                )
            for batch_id, links in urls.items():
                sys.stdout.write("\n  ".join([batch_id, *links]) + "\n")
            return
        from mlflow import MlflowClient

        from .loading import load_files
        from .rds_schema import create_tables

        summary = rebuild(
            settings,
            dict(os.environ),
            Path(args.data_dir),
            engine=create_db_engine(settings),
            client=MlflowClient(),
            load_files=load_files,
            create_tables=create_tables,
        )
        sys.stdout.write(f"{summary}\n")
    except RehearsalRefused as error:
        sys.exit(f"Refused: {error}")


if __name__ == "__main__":
    main()
