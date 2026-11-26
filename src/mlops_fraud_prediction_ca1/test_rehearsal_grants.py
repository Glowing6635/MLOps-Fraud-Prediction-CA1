import pytest

from .rehearsal_grants import reader_grant_sql
from .rehearsal_reset import RehearsalRefused


def test_generates_select_only_grants_on_the_rehearsal_database():
    assert reader_grant_sql("rehearsal", ["richard", "spiralsky"], {}) == [
        "GRANT SELECT ON `rehearsal`.* TO 'richard_reader'@'%'",
        "GRANT SELECT ON `rehearsal`.* TO 'spiralsky_reader'@'%'",
    ]


def test_output_is_deterministic_so_rerunning_is_idempotent():
    first = reader_grant_sql("rehearsal", ["a"], {})
    assert first == reader_grant_sql("rehearsal", ["a"], {})
    assert all(s.startswith("GRANT SELECT ON ") for s in first)
    assert not any(w in " ".join(first) for w in ("CREATE", "ALTER", "IDENTIFIED"))


@pytest.mark.parametrize(
    "database, environ",
    [
        ("fraud", {}),
        ("fraud", {"TEAM_DB_NAME": "fraud"}),
        ("rehearsal", {"TEAM_DB_NAME": "rehearsal"}),
    ],
)
def test_refuses_team_and_non_rehearsal_databases(database, environ):
    with pytest.raises(RehearsalRefused):
        reader_grant_sql(database, ["richard"], environ)


@pytest.mark.parametrize(
    "name", ["Richard", "x'; DROP USER a;--", "1abc", "a" * 17, ""]
)
def test_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        reader_grant_sql("rehearsal", [name], {})


def test_requires_a_name_and_a_plain_database_identifier():
    with pytest.raises(ValueError):
        reader_grant_sql("rehearsal", [], {})
    with pytest.raises(ValueError):
        reader_grant_sql("rehearsal`; x", ["a"], {})
