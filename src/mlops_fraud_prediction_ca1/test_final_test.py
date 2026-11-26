"""Tests for the one-shot final test harness (PBI #23), on made-up data only.

No real data file is read. The temporary split configuration and the temporary git
repository stand in for the project's own, and the protected final window is never
opened: the one test that reaches the protected reader uses a made-up folder.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from mlops_fraud_prediction_ca1 import final_test
from mlops_fraud_prediction_ca1.final_test import (
    AS_TRAINED,
    FREEZE_FILE,
    REFIT,
    FinalInputs,
    FinalTestRefused,
    Window,
    main,
    run_final_test,
    verify_freeze,
)

K = 2
X = 100.0
LOYAL = 2
CONFIG = """train_start = "2022-11-12"
validation_start = "2023-03-27"
test_start = "2023-04-17"
test_end_exclusive = "2023-05-11"
test_locked = true
final_evaluation_enabled = {enabled}
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository with one commit that freezes K, X and the loyal rule."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    root.joinpath(FREEZE_FILE).write_text(
        f"k = {K}\nx = {X}\nloyal_min_prior_orders = {LOYAL}\n"
    )
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "freeze")
    root.joinpath("later.txt").write_text("code after the freeze\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "later")
    return root


@pytest.fixture
def freeze(repo):
    return _git(repo, "rev-parse", "HEAD~1")


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "time_split.toml"
    path.write_text(CONFIG.format(enabled="false"))
    return path


class _ByColumn(ClassifierMixin, BaseEstimator):
    """Scores by the `signal` feature, and counts the rows it was fitted on."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fitted_rows_ = len(X)
        return self

    def predict_proba(self, X):
        score = X["signal"].to_numpy(dtype=float)
        return np.column_stack([1 - score, score])


def _window(rows):
    keys = pd.DataFrame(
        rows, columns=["order_date", "country_code", "order_id", "is_fraud"]
    )
    return keys


def make_inputs():
    """Two test days of five orders each, with scores and order values set by hand.

    Day 1: fraud A,B,C; the champion ranks A, D, B, ... the rule flags A and E.
    """
    test = pd.DataFrame(
        {
            "order_date": ["2023-04-18"] * 5 + ["2023-04-19"] * 5,
            "country_code": "MY",
            "order_id": list("ABCDEFGHIJ"),
            "is_fraud": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            "signal": [0.9, 0.7, 0.1, 0.8, 0.2, 0.95, 0.9, 0.1, 0.1, 0.1],
            "order_value": [500.0, 10, 10, 10, 300, 400, 10, 10, 10, 10],
        }
    )
    train = pd.DataFrame(
        {
            "order_date": "2022-12-01",
            "country_code": "MY",
            "order_id": [f"t{i}" for i in range(6)],
            "is_fraud": [0, 1, 0, 1, 0, 0],
            "signal": [0.1, 0.9, 0.2, 0.8, 0.1, 0.3],
        }
    )
    validation = train.assign(
        order_date="2023-04-01", order_id=lambda f: "v" + f.order_id
    )
    keys = ["order_date", "country_code", "order_id", "is_fraud"]
    # Orders in rule_orders are shuffled to prove alignment is by key, not position.
    shuffled = test.sample(frac=1, random_state=3)
    age = pd.Series(0, index=shuffled.index)
    age[shuffled.order_id.isin(["D"])] = 5  # D is an old account, so the rule skips it.
    return FinalInputs(
        train=Window(train[keys], train[["signal"]]),
        validation=Window(validation[keys], validation[["signal"]]),
        test=Window(test[keys], test[["signal"]]),
        rule_orders=shuffled[[*keys, "order_value"]],
        account_age=age.reset_index(drop=True),
        loyalty=np.array(["not_loyal"] * len(shuffled)),
    )


class Loader:
    """Counts reads, and can check what the ledger held when the data was read."""

    def __init__(self, ledger=None, boom=False):
        self.calls, self.seen, self.ledger, self.boom = 0, [], ledger, boom

    def __call__(self):
        self.calls += 1
        if self.ledger is not None and Path(self.ledger).exists():
            self.seen = Path(self.ledger).read_text().splitlines()
        if self.boom:
            raise RuntimeError("read failed")
        return make_inputs()


@pytest.fixture
def run(repo, freeze, config, tmp_path):
    ledger = tmp_path / "ledger.jsonl"

    def go(**overrides):
        arguments = {
            "freeze_commit": freeze,
            "k": K,
            "x": X,
            "loyal_min_prior_orders": LOYAL,
            "mode": AS_TRAINED,
            "champion": _ByColumn().fit(pd.DataFrame({"signal": [0.1]}), [0]),
            "champion_id": "made-up champion",
            "load_inputs": Loader(ledger),
            "ledger": ledger,
            "config_path": config,
            "repo": repo,
        }
        arguments.update(overrides)
        return run_final_test(**arguments)

    go.ledger = ledger
    return go


def _events(ledger):
    return [json.loads(line) for line in Path(ledger).read_text().splitlines()]


# --- freezing commit -------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_refuses_without_a_named_freezing_commit(run, missing):
    loader = Loader()
    with pytest.raises(FinalTestRefused, match="Name the freezing commit"):
        run(freeze_commit=missing, load_inputs=loader)
    assert loader.calls == 0
    assert not run.ledger.exists()


def test_refuses_an_unknown_commit(run):
    loader = Loader()
    with pytest.raises(FinalTestRefused):
        run(freeze_commit="deadbeef" * 5, load_inputs=loader)
    assert loader.calls == 0


def test_refuses_a_commit_that_is_not_an_ancestor_of_head(run, repo):
    _git(repo, "checkout", "-q", "-b", "side")
    repo.joinpath("side.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "side")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    with pytest.raises(FinalTestRefused, match="not an ancestor"):
        run(freeze_commit=side)


@pytest.mark.parametrize(
    ("name", "value"), [("k", K + 1), ("x", X + 1.0), ("loyal_min_prior_orders", 3)]
)
def test_refuses_values_the_commit_did_not_freeze(run, name, value):
    with pytest.raises(FinalTestRefused, match=f"{name} is"):
        run(**{name: value})
    assert not run.ledger.exists()


def test_refuses_a_commit_without_the_freeze_file(run, repo):
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD")
    _git(repo, "checkout", "-q", "-b", "bare", first)
    _git(repo, "rm", "-q", FREEZE_FILE)
    _git(repo, "commit", "-q", "-m", "no freeze")
    bare = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-edit", "bare")
    with pytest.raises(FinalTestRefused, match="failed"):
        run(freeze_commit=bare)


def test_refuses_a_loyal_definition_the_code_does_not_apply(repo, tmp_path):
    repo.joinpath(FREEZE_FILE).write_text(
        f"k = {K}\nx = {X}\nloyal_min_prior_orders = 5\n"
    )
    _git(repo, "commit", "-qam", "freeze a different loyal rule")
    sha = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(FinalTestRefused, match="not the one this code applies"):
        verify_freeze(sha, k=K, x=X, loyal_min_prior_orders=5, repo=repo)


def test_accepts_the_named_commit_and_returns_its_full_sha(repo, freeze):
    sha = verify_freeze(freeze[:10], k=K, x=X, loyal_min_prior_orders=LOYAL, repo=repo)
    assert sha == freeze


# --- champion mode ---------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "", "refit", "as_trained"])
def test_refuses_without_a_recorded_mode(run, mode):
    loader = Loader()
    with pytest.raises(FinalTestRefused, match="champion mode"):
        run(mode=mode, load_inputs=loader)
    assert loader.calls == 0
    assert not run.ledger.exists()


def test_mode_is_on_the_ledger_before_any_data_is_read(run):
    loader = Loader(run.ledger)
    run(mode=REFIT, load_inputs=loader)
    assert loader.calls == 1
    assert len(loader.seen) == 1
    claim = json.loads(loader.seen[0])
    assert claim["event"] == "mode_recorded"
    assert claim["mode"] == REFIT
    assert [e["event"] for e in _events(run.ledger)] == ["mode_recorded", "scored"]


def test_as_trained_scores_the_champion_unchanged(run):
    champion = _ByColumn().fit(pd.DataFrame({"signal": [0.1] * 3}), [0, 0, 1])
    entry = run(mode=AS_TRAINED, champion=champion)
    assert champion.fitted_rows_ == 3
    assert entry["mode"] == AS_TRAINED


def test_refit_fits_a_copy_on_train_and_validation_only(run):
    champion = _ByColumn().fit(pd.DataFrame({"signal": [0.1] * 3}), [0, 0, 1])
    seen = []
    original = final_test.prepare_champion

    def spy(model, mode, inputs):
        refit = original(model, mode, inputs)
        seen.append(refit)
        return refit

    final_test.prepare_champion = spy
    try:
        run(mode=REFIT, champion=champion)
    finally:
        final_test.prepare_champion = original
    assert champion.fitted_rows_ == 3  # the stored champion is untouched
    assert seen[0] is not champion
    assert seen[0].fitted_rows_ == 12  # 6 train + 6 validation, no test rows


def test_refit_keeps_the_stored_hyperparameters():
    champion = LogisticRegression(C=0.37, max_iter=50, random_state=7)
    features = pd.DataFrame({"signal": [0.0, 1.0, 0.1, 0.9]})
    champion.fit(features, [0, 1, 0, 1])
    inputs = make_inputs()
    refit = final_test.prepare_champion(champion, REFIT, inputs)
    assert refit.get_params() == champion.get_params()
    assert refit is not champion
    assert refit.n_features_in_ == 1


# --- one shot --------------------------------------------------------------------


def test_second_scoring_of_the_same_window_is_refused(run, repo, freeze):
    run()
    loader = Loader()
    with pytest.raises(FinalTestRefused, match="already used"):
        run(load_inputs=loader)
    assert loader.calls == 0
    assert [e["event"] for e in _events(run.ledger)] == ["mode_recorded", "scored"]


def test_a_different_freezing_commit_does_not_reopen_the_window(run, repo, freeze):
    run()
    repo.joinpath("again.txt").write_text("again")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "refreeze")
    with pytest.raises(FinalTestRefused, match="already used"):
        run(freeze_commit=_git(repo, "rev-parse", "HEAD"), mode=REFIT)


def test_a_failed_run_still_uses_the_window(run):
    with pytest.raises(RuntimeError, match="read failed"):
        run(load_inputs=Loader(boom=True))
    assert [e["event"] for e in _events(run.ledger)] == ["mode_recorded", "failed"]
    with pytest.raises(FinalTestRefused, match="already used"):
        run()


def test_an_interrupted_claim_blocks_the_window(run):
    run.ledger.write_text(
        json.dumps({"event": "mode_recorded", "window": "2023-04-17/2023-05-11"}) + "\n"
    )
    with pytest.raises(FinalTestRefused, match="already used"):
        run()


def test_another_window_in_the_ledger_does_not_block(run):
    run.ledger.write_text(
        json.dumps({"event": "scored", "window": "2030-01-01/2030-02-01"}) + "\n"
    )
    assert run()["event"] == "scored"


# --- dry run ---------------------------------------------------------------------


def test_dry_run_checks_everything_and_touches_nothing(run):
    loader = Loader()
    champion = object()  # a dry run never fits or scores the champion
    entry = run(dry_run=True, load_inputs=loader, champion=champion)
    assert entry["dry_run"] is True
    assert entry["mode"] == AS_TRAINED
    assert loader.calls == 0
    assert not run.ledger.exists()


def test_dry_run_still_refuses_a_used_window(run):
    run()
    with pytest.raises(FinalTestRefused, match="already used"):
        run(dry_run=True)


def test_dry_run_still_refuses_a_bad_freeze(run):
    with pytest.raises(FinalTestRefused):
        run(dry_run=True, freeze_commit=None)


def test_a_real_run_refuses_a_dirty_tree(run, repo):
    repo.joinpath("later.txt").write_text("edited")
    loader = Loader()
    with pytest.raises(FinalTestRefused, match="uncommitted"):
        run(load_inputs=loader)
    assert loader.calls == 0
    assert not run.ledger.exists()


# --- logged measures --------------------------------------------------------------


def test_logs_primary_measure_precision_and_recall_for_champion_and_rule(run):
    entry = run()
    result = entry["result"]
    assert result["primary_measure"] == "fraud caught at K"
    assert (result["orders"], result["fraud"], result["k"]) == (10, 4, 2)
    # Champion, K=2 a day. Day 1 top two by signal: A (fraud), D. Day 2: F (fraud), G.
    champion = result["champion"]
    assert champion["fraud_caught_at_k"] == 2
    assert champion["flags"] == 4
    assert champion["precision_at_k"] == 0.5
    assert champion["recall_at_k"] == 0.5
    # Rule: value > 100 and age 0. Day 1 A, E; day 2 F. All three within K.
    rule = result["rule"]
    assert rule["flags"] == 3
    assert rule["fraud_caught_at_k"] == 2
    assert rule["precision_at_k"] == pytest.approx(2 / 3)
    assert rule["recall_at_k"] == 0.5
    assert entry["freeze_commit"] and entry["code_commit"]
    # The ledger holds exactly what the command reports.
    assert _events(run.ledger)[-1] == entry


def test_scores_align_to_the_rule_orders_by_key_not_position(run):
    shuffled = run()["result"]
    assert shuffled["champion"]["fraud_caught_at_k"] == 2


def test_refuses_when_test_features_and_rule_orders_differ(run):
    def bad():
        inputs = make_inputs()
        return FinalInputs(
            **{
                **inputs.__dict__,
                "rule_orders": inputs.rule_orders.iloc[:-1],
                "account_age": inputs.account_age.iloc[:-1],
                "loyalty": inputs.loyalty[:-1],
            }
        )

    with pytest.raises(ValueError, match="different orders"):
        run(load_inputs=bad)


def test_logs_to_mlflow_when_asked(run, tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "final-test-synthetic")
    run(log_mlflow=final_test.log_to_mlflow)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["final-test-synthetic"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["metrics.champion_fraud_caught_at_k"] == 2
    assert row["metrics.rule_precision_at_k"] == pytest.approx(2 / 3)
    assert row["params.mode"] == AS_TRAINED


# --- command line ----------------------------------------------------------------


def _cli(repo, freeze, config, ledger, *extra):
    return [
        "--freeze-commit", freeze, "--k", str(K), "--x", str(X),
        "--loyal-min-prior-orders", str(LOYAL), "--champion-mode", AS_TRAINED,
        "--champion-uri", "models:/made-up@champion", "--config", str(config),
        "--ledger", str(ledger), *extra,
    ]  # fmt: skip


def test_command_dry_run(repo, freeze, config, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    ledger = tmp_path / "l.jsonl"
    main(_cli(repo, freeze, config, ledger, "--dry-run"))
    assert "DRY RUN" in capsys.readouterr().out
    assert not ledger.exists()


def test_command_exits_with_the_refusal(repo, freeze, config, tmp_path, monkeypatch):
    monkeypatch.chdir(repo)
    argv = _cli(repo, freeze, config, tmp_path / "l.jsonl", "--dry-run")
    argv[argv.index("--k") + 1] = "9"
    with pytest.raises(SystemExit, match="final-test refused"):
        main(argv)


@pytest.mark.parametrize("flag", ["--freeze-commit", "--champion-mode", "--k", "--x"])
def test_command_requires_each_frozen_input(repo, freeze, config, tmp_path, flag):
    argv = _cli(repo, freeze, config, tmp_path / "l.jsonl", "--dry-run")
    position = argv.index(flag)
    del argv[position : position + 2]
    with pytest.raises(SystemExit):
        main(argv)


# --- the protected reader stays closed --------------------------------------------


@pytest.fixture
def made_up_dir(tmp_path):
    """Made-up base CSVs with dates inside the real windows; no real file is read."""
    rng = np.random.default_rng(0)
    customers = pd.DataFrame(
        {
            "country_code": "MY",
            "customer_id": [f"c{i}" for i in range(40)],
            "mobile_verified": "true",
            "num_orders_last_50days": 1,
            "num_cancelled_orders_last_50days": 0,
            "num_refund_orders_last_50days": 0,
            "total_payment_last_50days": 10.0,
            "num_associated_customers": 1,
            "first_order_datetime": "2022-01-01 10:00:00",
        }
    )
    orders, labels = [], []
    for window, (start, count) in {
        "train": ("2022-11-20", 60),
        "validation": ("2023-03-28", 30),
        "test": ("2023-04-18", 30),
    }.items():
        for i in range(count):
            order_id = f"{window}{i}"
            fraud = int(i % 5 == 0)
            orders.append(
                {
                    "country_code": "MY",
                    "order_id": order_id,
                    "collect_type": "delivery",
                    "payment_method": "credit card",
                    "order_value": float(rng.integers(10, 200)),
                    "num_items_ordered": 2,
                    "refund_value": 0,
                    "order_date": str(
                        (pd.Timestamp(start) + pd.Timedelta(days=i % 5)).date()
                    ),
                }
            )
            labels.append(
                {
                    "country_code": "MY",
                    "order_id": order_id,
                    "customer_id": f"c{i % 40}",
                    "is_fraud": fraud,
                }
            )
    for name, frame in (
        ("customers", customers),
        ("orders", pd.DataFrame(orders)),
        ("labels", pd.DataFrame(labels)),
    ):
        frame.to_csv(tmp_path / f"{name}_base.csv", index=False)
    return tmp_path


def test_real_inputs_stay_closed_under_the_default_configuration(made_up_dir):
    """On a made-up folder: the shipped config keeps final evaluation disabled."""
    with pytest.raises(ValueError, match="final_evaluation_enabled"):
        final_test._real_inputs(made_up_dir, None)


def test_real_inputs_read_made_up_files_through_the_reader(made_up_dir, tmp_path):
    """The loader path works end to end on made-up CSVs and a made-up open config."""
    config = tmp_path / "open.toml"
    config.write_text(CONFIG.format(enabled="true"))
    inputs = final_test._real_inputs(made_up_dir, config)
    assert len(inputs.train.keys) == 60
    assert len(inputs.validation.keys) == 30
    assert len(inputs.test.keys) == len(inputs.rule_orders) == 30
    assert set(inputs.test.keys.order_id) == set(inputs.rule_orders.order_id)
    assert len(inputs.account_age) == len(inputs.loyalty) == 30
