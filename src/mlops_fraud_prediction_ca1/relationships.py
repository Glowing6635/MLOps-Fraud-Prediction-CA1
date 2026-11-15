"""Nonmutating relationship audits with explicit keys and safe join cardinality."""

from collections.abc import Sequence

import pandas as pd


def _prepared(frame: pd.DataFrame, normalize_country: bool) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if normalize_country and "country_code" in result:
        result["country_code"] = result["country_code"].astype("string").str.strip().str.upper()
    return result


def _keys(frame: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Specify nonempty, distinct key columns")
    frame[keys]  # Fail clearly for absent columns.
    return keys


def key_profile(frame: pd.DataFrame, keys: Sequence[str], *, normalize_country: bool = False) -> dict:
    """Count key duplicates separately from full-row copies; null keys are excluded.

    ``duplicate_rows`` counts every row in repeated key groups; ``extra_key_rows``
    counts rows beyond one per key. Conflicts mean different full rows sharing a
    key after the optional country formatting. Uniqueness applies to complete
    keys only, and missing-key rows are reported explicitly.
    """
    keys = _keys(frame, keys)
    data = _prepared(frame, normalize_country)
    valid = data.loc[data[keys].notna().all(axis=1)]
    sizes = valid.groupby(keys, dropna=False).size()
    distinct = valid.drop_duplicates()
    conflicts = distinct.duplicated(keys, keep=False)
    return {
        "rows": len(data), "null_key_rows": len(data) - len(valid),
        "complete_key_rows": len(valid), "distinct_keys": len(sizes),
        "duplicate_groups": int(sizes.gt(1).sum()),
        "duplicate_rows": int(sizes.loc[sizes.gt(1)].sum()),
        "extra_key_rows": int((sizes - 1).sum()),
        "exact_duplicate_rows": int(data.duplicated(keep=False).sum()),
        "exact_extra_copies": int(data.duplicated().sum()),
        "conflicting_keys": len(distinct.loc[conflicts, keys].drop_duplicates()),
        "unique_complete_keys": not bool(sizes.gt(1).any()),
    }


def unique_records(frame: pd.DataFrame, keys: Sequence[str], *, normalize_country: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return one row per unambiguous complete key and a loss/exclusion audit.

    Exact full-row copies are collapsed; all differing rows for a conflicting
    key are excluded, never arbitrarily selected. Country formatting can also
    collapse rows identical after formatting; both copy counts are disclosed.
    """
    keys = _keys(frame, keys)
    data = _prepared(frame, normalize_country)
    audit = key_profile(frame, keys, normalize_country=normalize_country)
    audit["raw_exact_extra_copies"] = int(frame.duplicated().sum())
    complete = data.loc[data[keys].notna().all(axis=1)]
    distinct = complete.drop_duplicates()
    conflict = distinct.duplicated(keys, keep=False)
    conflict_keys = distinct.loc[conflict, keys].drop_duplicates()
    audit["conflicting_source_rows"] = int(
        pd.MultiIndex.from_frame(complete[keys]).isin(pd.MultiIndex.from_frame(conflict_keys)).sum()
    )
    result = distinct.loc[~conflict].copy()
    audit["safe_rows"] = len(result)
    return result, audit


def bidirectional_matches(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str], *, normalize_country: bool = False) -> pd.DataFrame:
    """Report both row-weighted and distinct-key coverage without joining rows.

    Rates divide by complete-key rows/keys, respectively; null-key rows never
    match and are reported separately. Duplicate rows retain their row weight.
    These are existence checks, not proof of unambiguous mapping.
    """
    keys = _keys(left, keys)
    _keys(right, keys)
    frames = [_prepared(left, normalize_country), _prepared(right, normalize_country)]
    complete = [frame.loc[frame[keys].notna().all(axis=1), keys] for frame in frames]
    indexes = [pd.MultiIndex.from_frame(frame) for frame in complete]
    records = []
    for i, name in enumerate(("left_to_right", "right_to_left")):
        matched = int(indexes[i].isin(indexes[1 - i]).sum())
        unique = indexes[i].unique()
        matched_unique = int(unique.isin(indexes[1 - i]).sum())
        records.append({"direction": name, "rows": len(frames[i]),
                        "null_key_rows": len(frames[i]) - len(indexes[i]),
                        "complete_key_rows": len(indexes[i]), "matched_rows": matched,
                        "unmatched_rows": len(indexes[i]) - matched,
                        "row_match_rate": matched / len(indexes[i]) if len(indexes[i]) else float("nan"),
                        "distinct_keys": len(unique), "matched_distinct_keys": matched_unique,
                        "key_match_rate": matched_unique / len(unique) if len(unique) else float("nan")})
    return pd.DataFrame(records).set_index("direction")


def country_agreement(left: pd.DataFrame, right: pd.DataFrame, id_column: str, *, country_column: str = "country_code", normalize_country: bool = False) -> pd.DataFrame:
    """Compare country sets per shared nonnull ID, avoiding Cartesian joins.

    Any missing country makes the comparison incomplete (``missing_country``).
    An ID is ``ambiguous``
    when either side has multiple observed countries, even if the sets agree.
    Only shared IDs appear; use bidirectional_matches for absent IDs.
    """
    def pairs(frame):
        data = _prepared(frame, normalize_country)
        if normalize_country and country_column != "country_code":
            data[country_column] = data[country_column].astype("string").str.strip().str.upper()
        return data.loc[data[id_column].notna(), [id_column, country_column]].drop_duplicates()
    a, b = pairs(left), pairs(right)
    ac = a.groupby(id_column)[country_column].nunique().rename("left_country_count")
    bc = b.groupby(id_column)[country_column].nunique().rename("right_country_count")
    result = pd.concat([ac, bc], axis=1, join="inner")
    result["left_missing_country"] = result.index.isin(a.loc[a[country_column].isna(), id_column])
    result["right_missing_country"] = result.index.isin(b.loc[b[country_column].isna(), id_column])
    incomplete = result.left_missing_country | result.right_missing_country
    complete_a, complete_b = a.dropna(), b.dropna()
    shared = complete_a.loc[pd.MultiIndex.from_frame(complete_a).isin(pd.MultiIndex.from_frame(complete_b))]
    result["shared_country_count"] = shared.groupby(id_column).size().reindex(result.index, fill_value=0)
    result["sets_equal"] = (result.left_country_count.eq(result.right_country_count)
                            & result.shared_country_count.eq(result.left_country_count)
                            & result.left_country_count.gt(0) & ~incomplete)
    result["status"] = "disagree"
    result.loc[result.sets_equal, "status"] = "agree"
    result.loc[result.left_country_count.gt(1) | result.right_country_count.gt(1), "status"] = "ambiguous"
    result.loc[incomplete, "status"] = "missing_country"
    return result.reset_index()


def link_orders_customers(orders: pd.DataFrame, labels: pd.DataFrame, customers: pd.DataFrame, *, normalize_country: bool = False) -> tuple[pd.DataFrame, dict]:
    """Bridge safe orders to customers through safe labels using country + ID.

    Returns every safe order once, with ``label_link``/``customer_link`` merge
    indicators and suffixed payload columns where names overlap. A missing or
    conflicting label cannot supply customer or fraud data. Audit dictionaries
    expose all source exclusions. No source is changed; joins validate cardinality.
    """
    order_keys, customer_keys = ["country_code", "order_id"], ["country_code", "customer_id"]
    o, oa = unique_records(orders, order_keys, normalize_country=normalize_country)
    l, la = unique_records(labels, order_keys, normalize_country=normalize_country)
    c, ca = unique_records(customers, customer_keys, normalize_country=normalize_country)
    joined = o.merge(l, on=order_keys, how="left", validate="one_to_one", indicator="label_link", suffixes=("", "_label"))
    joined = joined.merge(c, on=customer_keys, how="left", validate="many_to_one", indicator="customer_link", suffixes=("", "_customer"))
    return joined, {"orders": oa, "labels": la, "customers": ca,
                    "safe_orders": len(o), "orders_with_label": int(joined.label_link.eq("both").sum()),
                    "orders_with_customer": int(joined.customer_link.eq("both").sum())}
