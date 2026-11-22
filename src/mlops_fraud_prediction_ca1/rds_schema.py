"""Set up the team database: its tables, and its users (issues #10 and #20).

- `uv run rds-schema tables` creates every table in `schema.sql`.
- `uv run rds-schema reader <name>` creates `<name>_reader`, which can only read the
  database, or changes its password if it exists. `writer` does the same for
  `<name>_writer`, which can also add rows, as `load-batch` needs, but never change
  or delete them. Both ask for the password, so none is typed on the command line
  or kept in git.
- `uv run rds-schema check` lists the tables and grants the configured user sees.

Every command connects with the settings in `.env`. `tables`, `reader` and `writer`
need the instance's admin user, and all three are safe to re-run.
"""

import argparse
import getpass
import re
from importlib.resources import files

from sqlalchemy import Engine, text

from mlops_fraud_prediction_ca1.config import create_db_engine, load_settings

# Names go into MySQL statements, so allow only plain lowercase words.
_NAME = re.compile(r"[a-z][a-z0-9_]{0,15}")
_DATABASE = re.compile(r"\w+")
MIN_PASSWORD_LENGTH = 12
# A writer only adds rows: the loader never changes or deletes a stored row.
_GRANTS = {"reader": "SELECT", "writer": "SELECT, INSERT"}


def statements() -> list[str]:
    """The statements in `schema.sql`, without its comments."""
    sql = files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    return [statement.strip() for statement in body.split(";") if statement.strip()]


def create_tables(engine: Engine) -> None:
    """Create any table that does not exist yet; existing tables are left alone."""
    with engine.begin() as connection:
        for statement in statements():
            connection.execute(text(statement))


def create_user(
    engine: Engine, database: str, name: str, password: str, role: str
) -> str:
    """Create or update `<name>_<role>` with the role's grants on `database`, and return its name."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            "The name must be lowercase letters, digits or _, start with a letter,"
            " and have at most 16 characters."
        )
    if not _DATABASE.fullmatch(database):
        raise ValueError(f"Unexpected database name {database!r}.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Use a password of at least {MIN_PASSWORD_LENGTH} characters."
        )
    user = f"{name}_{role}"
    login = {"user": user, "password": password}
    with engine.begin() as connection:
        connection.execute(
            text("CREATE USER IF NOT EXISTS :user@'%' IDENTIFIED BY :password"), login
        )
        connection.execute(text("ALTER USER :user@'%' IDENTIFIED BY :password"), login)
        connection.execute(
            text(f"GRANT {_GRANTS[role]} ON `{database}`.* TO :user@'%'"),
            {"user": user},
        )
    return user


def check(engine: Engine) -> tuple[list[str], list[str]]:
    """The tables the configured user can see, and that user's grants."""
    with engine.connect() as connection:
        tables = list(connection.execute(text("SHOW TABLES")).scalars())
        grants = list(connection.execute(text("SHOW GRANTS")).scalars())
    return tables, grants


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rds-schema", description="Set up the team database (issue #10)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tables", help="create the tables; safe to re-run")
    for role, help_text in [
        ("reader", "create or update an engineer's read-only user"),
        ("writer", "create or update the user that loads batches"),
    ]:
        user = commands.add_parser(role, help=help_text)
        user.add_argument("name", help="the engineer's short name, such as richard")
    commands.add_parser("check", help="list the tables and grants you can see")
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = create_db_engine(settings)
    if args.command == "tables":
        create_tables(engine)
        print(f"Tables ready in {settings.db_name} on {settings.db_host}.")
    elif args.command in _GRANTS:
        user = f"{args.name}_{args.command}"
        password = getpass.getpass(f"Password for {user}: ")
        create_user(engine, settings.db_name, args.name, password, args.command)
        print(
            f"{user} has {_GRANTS[args.command]} on {settings.db_name}."
            " Keep the password out of git."
        )
    else:
        tables, grants = check(engine)
        print(
            f"Connected to {settings.db_name} on {settings.db_host} as {settings.db_user}."
        )
        print("Tables: " + (", ".join(tables) or "none"))
        print("\n".join(grants))
