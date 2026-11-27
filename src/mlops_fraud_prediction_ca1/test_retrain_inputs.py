import hashlib
from types import SimpleNamespace

import pytest

from . import retrain_inputs

_VALIDATION = (
    "country_code,order_id,customer_id,order_date,is_fraud,"
    "collect_type,payment_method,order_value,num_items_ordered\n"
    "MY,001,009,2023-03-27,0,delivery,credit card,15.00,1\n"
    "MY,002,008,2023-04-16,1,delivery,credit card,,2\n"
)
_RUN_ID = "0123456789abcdef" * 2


@pytest.fixture
def validation(tmp_path, monkeypatch):
    path = tmp_path / "validation.csv"
    path.write_text(_VALIDATION, encoding="utf-8")
    for name, value in {
        "RETRAIN_VALIDATION_CSV": str(path),
        "RETRAIN_VALIDATION_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "RETRAIN_CHAMPION_MODEL_NAME": "fraud-candidate",
        "RETRAIN_CHAMPION_MODEL_VERSION": "1",
        "RETRAIN_CHAMPION_RUN_ID": _RUN_ID,
    }.items():
        monkeypatch.setenv(name, value)
    return path


@pytest.mark.parametrize(
    ("change", "update_digest", "error"),
    [
        ("2023-03-27", False, "does not match"),
        ("2023-03-27", True, "exclusively from"),
        ("MY,002", True, "unique labelled orders"),
    ],
)
def test_factory_rejects_invalid_validation_before_registry_lookup(
    validation, monkeypatch, change, update_digest, error
):
    replacement = "MY,001" if change.startswith("MY") else "2025-01-06"
    validation.write_text(_VALIDATION.replace(change, replacement), encoding="utf-8")
    if update_digest:
        monkeypatch.setenv(
            "RETRAIN_VALIDATION_SHA256",
            hashlib.sha256(validation.read_bytes()).hexdigest(),
        )

    def unreachable():
        pytest.fail("Validation rejection must precede registry access")

    monkeypatch.setattr(retrain_inputs, "MlflowClient", unreachable)
    with pytest.raises(ValueError, match=error):
        retrain_inputs.build_inputs()


def test_factory_rejects_champion_version_from_another_run(validation, monkeypatch):
    def get_model_version(name, version):
        assert (name, version) == ("fraud-candidate", "1")
        return SimpleNamespace(run_id="f" * 32)

    monkeypatch.setattr(
        retrain_inputs,
        "MlflowClient",
        lambda: SimpleNamespace(get_model_version=get_model_version),
    )
    with pytest.raises(ValueError, match="different MLflow run"):
        retrain_inputs.build_inputs()
