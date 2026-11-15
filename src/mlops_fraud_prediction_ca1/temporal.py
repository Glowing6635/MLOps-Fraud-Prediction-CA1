"""Explicit date parsing and descriptive fraud rates; source rows stay intact."""

from collections.abc import Sequence
from numbers import Real

import pandas as pd


_MONTHS = {name: f"{i:02}" for i, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def parse_dates(values: pd.Series, *, kind: str = "order") -> pd.DataFrame:
    """Return original/date/status/format using only known explicit formats.

    Order formats: YYYY/MM/DD, YYYY-MM-DD, YYYY/Mon/DD (English months).
    Customer format: YYYY-MM-DD HH:MM:SS. Missing values, unrecognized shapes,
    and recognized shapes with invalid dates are separate. Blank strings are
    unrecognized. No timezone is assumed and no input is modified/coerced.
    """
    if kind not in {"order", "customer"}:
        raise ValueError("kind must be 'order' or 'customer'")
    result = pd.DataFrame({"original": values, "date": pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]"),
                           "status": "unrecognized", "format": pd.Series(None, index=values.index, dtype=object)})
    text = values.astype("string")
    string_values = values.map(lambda value: isinstance(value, str))
    formats = [(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S")] if kind == "customer" else [
        (r"\d{4}/\d{2}/\d{2}", "%Y/%m/%d"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{4}/[A-Za-z]{3}/\d{2}", "%Y/%b/%d")]
    for pattern, fmt in formats:
        mask = string_values & text.str.fullmatch(pattern, na=False)
        selected = text.loc[mask]
        parse_format = fmt
        if fmt == "%Y/%b/%d":
            month = selected.str.slice(5, 8).map(_MONTHS).fillna("00")
            selected = selected.str.slice(0, 5) + month + selected.str.slice(8)
            parse_format = "%Y/%m/%d"
        parsed = pd.to_datetime(selected, format=parse_format, errors="coerce", exact=True)
        result.loc[mask, "date"] = parsed
        result.loc[mask, "format"] = fmt
        result.loc[mask, "status"] = parsed.notna().map({True: "valid", False: "invalid"})
    result.loc[values.isna(), "status"] = "missing"
    return result


def date_profile(values: pd.Series, *, kind: str = "order") -> dict:
    """Report format/status counts and the valid date range, retaining denominators."""
    parsed = parse_dates(values, kind=kind)
    counts = parsed.status.value_counts()
    return {"rows": len(parsed), **{name: int(counts.get(name, 0)) for name in
            ("valid", "missing", "unrecognized", "invalid")},
            "minimum": parsed.date.min(), "maximum": parsed.date.max(),
            "formats": parsed["format"].value_counts().to_dict()}


def _validate_entities(rows: pd.DataFrame, entity_keys: Sequence[str]) -> None:
    keys = list(entity_keys)
    if not keys:
        raise ValueError("entity_keys must identify the disambiguated observations")
    if rows[keys].isna().any(axis=None) or rows.duplicated(keys).any():
        raise ValueError("Rates require unique complete entity keys; disambiguate rows first")


def fraud_rates(rows: pd.DataFrame, group_by: Sequence[str], *, label_column: str = "is_fraud", entity_keys: Sequence[str] = ("country_code", "order_id")) -> pd.DataFrame:
    """Rates over unique complete entities; duplicates/conflicts raise ValueError.

    Numeric 0/1 and literal strings '0'/'1' are valid; Boolean labels and all
    other values are invalid. Missing and invalid labels are excluded from the
    rate denominator and counted separately. Missing group values are retained.
    Call on a disambiguated join (e.g. link_orders_customers' result).
    """
    _validate_entities(rows, entity_keys)
    groups = list(group_by)
    if not groups:
        raise ValueError("Specify at least one grouping column")
    data = rows[groups].copy()
    values = rows[label_column]
    missing = values.isna()
    valid = values.map(lambda x: not isinstance(x, bool) and isinstance(x, (str, Real)) and x in (0, 1, "0", "1"))
    data["rows"] = 1
    data["valid_labels"] = valid.astype(int)
    data["fraud_labels"] = (valid & values.isin([1, "1"])).astype(int)
    data["missing_labels"] = missing.astype(int)
    data["invalid_labels"] = (~missing & ~valid).astype(int)
    result = data.groupby(groups, dropna=False, as_index=False).sum()
    result["fraud_rate"] = result.fraud_labels.div(result.valid_labels.where(result.valid_labels.ne(0)))
    return result


def weekly_fraud_rates(rows: pd.DataFrame, *, date_column: str = "order_date", country_column: str = "country_code", entity_keys: Sequence[str] = ("country_code", "order_id")) -> pd.DataFrame:
    """Monday–Sunday country rates; date exclusions are recorded in attrs.

    Partial weeks flag global dataset date boundaries, not an assertion that
    sampling is complete on other dates. Invalid/missing dates cannot be timed.
    Result attrs contain date_profile and excluded_date_rows for disclosure.
    """
    _validate_entities(rows, entity_keys)
    parsed = parse_dates(rows[date_column])
    data = rows.copy()
    data["week_start"] = parsed.date.dt.to_period("W-SUN").dt.start_time
    timed = data.loc[parsed.status.eq("valid")]
    result = fraud_rates(timed, [country_column, "week_start"], entity_keys=entity_keys)
    result["week_end"] = result.week_start + pd.Timedelta(days=6)
    minimum, maximum = parsed.date.min(), parsed.date.max()
    result["partial_week"] = result.week_start.lt(minimum) | result.week_end.gt(maximum)
    result.attrs["date_profile"] = date_profile(rows[date_column])
    result.attrs["excluded_date_rows"] = int(parsed.status.ne("valid").sum())
    return result.sort_values([country_column, "week_start"]).reset_index(drop=True)


def largest_weekly_change(rates: pd.DataFrame, *, country_column: str = "country_code") -> pd.DataFrame:
    """Largest absolute adjacent-week change per country, with counts/cautions.

    A descriptive candidate only, not a statistically established change point.
    Missing weeks, undefined rates, and zero-valid-label weeks are not compared.
    Ties use the earliest current week. Partial boundary weeks remain flagged.
    """
    columns = [country_column, "previous_week", "week_start", "previous_valid_labels", "valid_labels",
               "previous_fraud_labels", "fraud_labels", "previous_rate", "fraud_rate", "rate_change", "partial_week"]
    records = []
    for country, group in rates.groupby(country_column, dropna=False, sort=False):
        group = group.sort_values("week_start").reset_index(drop=True)
        if group.week_start.duplicated().any():
            raise ValueError("Provide one rate row per country and week")
        previous = group.shift()
        eligible = group.week_start.sub(previous.week_start).eq(pd.Timedelta(days=7)) & group.fraud_rate.notna() & previous.fraud_rate.notna()
        candidates = group.loc[eligible]
        if candidates.empty:
            continue
        changes = group.fraud_rate - previous.fraud_rate
        index = changes.loc[eligible].abs().idxmax()
        row, old = group.loc[index], previous.loc[index]
        records.append({country_column: country, "previous_week": old.week_start, "week_start": row.week_start,
                        "previous_valid_labels": int(old.valid_labels), "valid_labels": int(row.valid_labels),
                        "previous_fraud_labels": int(old.fraud_labels), "fraud_labels": int(row.fraud_labels),
                        "previous_rate": old.fraud_rate, "fraud_rate": row.fraud_rate,
                        "rate_change": changes.loc[index], "partial_week": bool(row.partial_week or old.partial_week)})
    return pd.DataFrame(records, columns=columns)
