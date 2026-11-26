"""Give each engineer's read-only user SELECT on the rehearsal database (PBI #24, AC1).

`uv run rehearsal-grant richard spiralsky` prints the statements. Add `--apply` to run
them with the admin login in `.env`. The grants are SELECT only, name the rehearsal
database and nothing else, and refuse any name that is not a rehearsal one or that is
the team's (`TEAM_DB_NAME`). A GRANT is idempotent, so re-running changes nothing. The
users must already exist (`rds-schema reader <name>`): this never creates a user or
changes a password, so it cannot touch the team database.
"""

import argparse
import sys
from collections.abc import Sequence
from types import SimpleNamespace

from .rds_schema import _DATABASE, _NAME
from .rehearsal_reset import RehearsalRefused, check_rehearsal_target


def reader_grant_sql(
    database: str, names: Sequence[str], environ: dict[str, str]
) -> list[str]:
    """`GRANT SELECT` statements on the rehearsal `database` for each `<name>_reader`."""
    if not names:
        raise ValueError("Give at least one engineer name.")
    if not _DATABASE.fullmatch(database):
        raise ValueError(f"Unexpected database name {database!r}.")
    # Reuse the reset's refusal; the experiment is not used here, so pass a valid one.
    check_rehearsal_target(
        SimpleNamespace(db_name=database, mlflow_experiment_name="rehearsal"), environ
    )
    for name in names:
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"Engineer name {name!r} must be lowercase letters, digits or _,"
                " start with a letter, and have at most 16 characters."
            )
    return [f"GRANT SELECT ON `{database}`.* TO '{name}_reader'@'%'" for name in names]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="+", help="engineer short names, such as richard"
    )
    parser.add_argument(
        "--apply", action="store_true", help="run the statements (default: print them)"
    )
    args = parser.parse_args(argv)

    import os

    from .config import load_settings

    settings = load_settings()
    try:
        statements = reader_grant_sql(settings.db_name, args.names, dict(os.environ))
    except (RehearsalRefused, ValueError) as error:
        sys.exit(f"Refused: {error}")
    if not args.apply:
        sys.stdout.write(";\n".join(statements) + ";\n")
        return
    from sqlalchemy import text

    from .config import create_db_engine

    with create_db_engine(settings).begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
    sys.stdout.write(
        f"Granted SELECT on {settings.db_name} to {len(statements)} user(s).\n"
    )
