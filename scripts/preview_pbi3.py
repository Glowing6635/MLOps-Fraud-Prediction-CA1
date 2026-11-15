"""Print PBI #3 evidence for review without modifying the notebook or CSVs.

Run from the repository: .venv/bin/python -B scripts/preview_pbi3.py
Use --section to inspect one criterion or --format json for machine-readable data.
Counts use source rows unless labelled as unique safe orders/keys. Rates are
fractions (0–1). The first and last observed weeks can be partial.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from mlops_fraud_prediction_ca1.data import load_base_data
from mlops_fraud_prediction_ca1.logic import DEFAULT_NUMERIC_COLUMNS, business_logic_report
from mlops_fraud_prediction_ca1.numeric import profile_numeric_features
from mlops_fraud_prediction_ca1.quality import CATEGORY_COLUMNS, quality_reports, raw_category_reports
from mlops_fraud_prediction_ca1.relationships import (
    bidirectional_matches, country_agreement, key_profile, link_orders_customers,
)
from mlops_fraud_prediction_ca1.temporal import (
    date_profile, fraud_rates, largest_weekly_change, weekly_fraud_rates,
)


HANDLING_PROPOSALS = (
    ("quality", "missing first-order timestamp", "Retain missing; exclude from date-order comparison, and disclose exclusions. Do not invent a first-order date."),
    ("quality", "missing payment method", "Retain missing as an explicit unknown category in later preparation; do not assume a payment method."),
    ("quality", "missing or invalid item count", "Retain nullable count and a missing/invalid flag; fit any later imputation on training data only."),
    ("quality", "exact duplicate rows", "Remove no rows when no copies exist. If copies arrive, collapse identical full rows only and audit their count."),
    ("quality", "types", "Keep IDs as text and money as decimal-valued numbers. Convert finite integral counts to nullable integers only in later cleaning; quarantine failed conversions."),
    ("quality", "mobile verification", "Accept specified True/False values; retain yes as unresolved with an invalid flag rather than infer its meaning."),
    ("quality", "category formats", "Preview uppercased countries and lowercased collection/payment values; retain originals. Preserve on Delivery without a semantic alias. Open category domains stay unchecked."),
    ("quality", "outliers and skew", "Retain raw values with IQR screening flags; do not cap or delete extremes without error evidence. Zero-IQR count flags do not establish errors. Evaluate any learned transforms within training only."),
    ("quality", "class imbalance", "Preserve observed prevalence. Evaluate fraud precision/recall and PR-AUC alongside a baseline; apply any weighting or resampling within training folds, never to validation/test data."),
    ("relationships", "keys and linkage", "Use country plus order_id, and link customers through labels. Collapse exact copies only; quarantine conflicting keys. Compare raw and normalized country results and disclose unlinked rows."),
    ("logic", "refund/count/date/negative flags", "Retain and flag rule violations for source confirmation; do not clamp refunds or remove possible fraud signals. Compare order and first-order dates by calendar day."),
    ("logic", "missing labels or customer", "Keep source rows visible in coverage reports. Exclude unlabelled orders from supervised targets and fraud-rate denominators; do not interpret missing labels as nonfraud. Mark customer features unavailable."),
    ("temporal", "date formats", "Parse only the three observed explicit order formats and the specified timestamp shape. Retain raw dates; isolate invalid/unrecognized values and leave missing timestamps missing."),
    ("temporal", "fraud trends", "Use valid safely linked labels, Monday-Sunday weeks and disclosed partial weeks. Date descriptive changes; do not claim a proven change point or cause from the largest weekly movement."),
)


def build_preview(data_dir: Path, section: str = "all") -> dict:
    """Return complete criterion reports; never write or repair source data.

    Relationship coverage compares raw keys with uppercase/trimmed country
    proposals. Temporal rates use unique normalized order keys and safe labels;
    customer linkage is disclosed but not required for a labelled order's rate.
    Conflicting records are excluded from joins, not resolved by keep-first.
    """
    if section not in {"all", "quality", "relationships", "logic", "temporal"}:
        raise ValueError("Unknown preview section")
    customers, orders, labels = load_base_data(data_dir)
    datasets = {"customers": customers, "orders": orders, "labels": labels}
    reports = {}
    if section in {"all", "quality"}:
        reports["quality"] = quality_reports(datasets)
        raw = {name: pd.read_csv(data_dir / f"{name}_base.csv", usecols=list(columns),
                                 dtype="string", keep_default_na=False)
               for name, columns in CATEGORY_COLUMNS.items()}
        reports["quality"]["raw_categories"] = raw_category_reports(raw)
        # Coerce only temporary statistical inputs. The validity report retains
        # source missingness and conversion failures as distinct evidence.
        numeric_inputs = {
            name: pd.DataFrame({column: pd.to_numeric(datasets[name][column], errors="coerce")
                                for column in columns})
            for name, columns in DEFAULT_NUMERIC_COLUMNS.items()
        }
        distributions = profile_numeric_features(numeric_inputs, DEFAULT_NUMERIC_COLUMNS).reset_index()
        distributions = distributions.rename(columns={"missing_count": "missing_or_unparseable_count"})
        reports["quality"]["numeric_distributions"] = distributions.merge(
            reports["quality"]["numeric_validity"][["dataset", "column", "missing", "conversion_failures"]]
            .rename(columns={"column": "feature", "missing": "source_missing_count"}),
            on=["dataset", "feature"], validate="one_to_one",
        )
    if section in {"all", "relationships"}:
        profiles, matches, agreements = [], [], []
        for normalized in (False, True):
            mode = "normalized_country" if normalized else "raw"
            for name, frame in datasets.items():
                for identifier in ("order_id", "customer_id"):
                    if identifier not in frame:
                        continue
                    for keys in ([identifier], ["country_code", identifier]):
                        profiles.append({"mode": mode, "dataset": name, "keys": "+".join(keys),
                                         **key_profile(frame, keys, normalize_country=normalized)})
            for left, right, identifier in (("orders", "labels", "order_id"),
                                            ("labels", "customers", "customer_id")):
                for keys in ([identifier], ["country_code", identifier]):
                    coverage = bidirectional_matches(datasets[left], datasets[right], keys,
                                                      normalize_country=normalized).reset_index()
                    coverage.insert(0, "relationship", f"{left} <-> {right}")
                    coverage.insert(0, "keys", "+".join(keys))
                    coverage.insert(0, "mode", mode)
                    matches.append(coverage)
                agreement = country_agreement(datasets[left], datasets[right], identifier,
                                              normalize_country=normalized)
                statuses = agreement.status.value_counts()
                agreements.append({"mode": mode, "relationship": f"{left} <-> {right}",
                                   "shared_ids": len(agreement),
                                   "unequal_country_sets": int((~agreement.sets_equal).sum()),
                                   **{status: int(statuses.get(status, 0)) for status in
                                      ("agree", "disagree", "ambiguous", "missing_country")}})
        reports["relationships"] = {"keys": pd.DataFrame(profiles),
                                     "coverage": pd.concat(matches, ignore_index=True),
                                     "country_agreement": pd.DataFrame(agreements)}
    if section in {"all", "logic"}:
        reports["logic"] = business_logic_report(customers, orders, labels,
                                                  normalize_country=True).reset_index()
    if section in {"all", "relationships", "temporal"}:
        linked, audit = link_orders_customers(orders, labels, customers, normalize_country=True)
        reports["link_audit"] = audit
        if section in {"all", "temporal"}:
            weekly = weekly_fraud_rates(linked)
            totals = ["rows", "valid_labels", "fraud_labels", "missing_labels", "invalid_labels"]
            overall = weekly.groupby("week_start", as_index=False)[totals].sum()
            overall["country_code"] = "ALL"
            overall["week_end"] = overall.week_start + pd.Timedelta(days=6)
            overall["partial_week"] = overall.week_start.isin(weekly.loc[weekly.partial_week, "week_start"])
            overall["fraud_rate"] = overall.fraud_labels.div(
                overall.valid_labels.where(overall.valid_labels.ne(0)))
            reports["temporal"] = {
                "order_dates": date_profile(orders.order_date),
                "customer_dates": date_profile(customers.first_order_datetime, kind="customer"),
                "country_rates": fraud_rates(linked, ["country_code"]),
                "weekly_country_rates": weekly,
                "weekly_overall_rates": overall,
                "largest_descriptive_weekly_changes": largest_weekly_change(
                    pd.concat([weekly, overall], ignore_index=True)),
                "excluded_date_rows": weekly.attrs["excluded_date_rows"],
                "change_caution": "Largest adjacent observed-week changes are descriptive candidates, not proven change points. Partial boundary weeks are flagged; absent weeks are not compared.",
            }
    reports["handling_proposals"] = pd.DataFrame(
        [row for row in HANDLING_PROPOSALS if section == "all" or row[0] == section],
        columns=["criterion", "topic", "proposed_handling"],
    )
    return reports


def _serializable(value):
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records", date_format="iso"))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _print_text(value, title="PBI #3 preview"):
    if isinstance(value, dict):
        for name, item in value.items():
            _print_text(item, f"{title} / {name}")
    else:
        print(f"\n{title}\n")
        print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--section", choices=["all", "quality", "relationships", "logic", "temporal"], default="all")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    reports = build_preview(args.data_dir, args.section)
    if args.format == "json":
        print(json.dumps(_serializable(reports), indent=2, allow_nan=False))
    else:
        _print_text(reports)


if __name__ == "__main__":
    main()
