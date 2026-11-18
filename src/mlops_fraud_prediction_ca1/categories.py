"""Preview category formatting and validation without changing source data."""

from collections.abc import Collection, Mapping
from typing import Literal

import pandas as pd


def standardise_categories(
    values: pd.Series,
    *,
    case: Literal["lower", "upper"] = "lower",
    strip: bool = True,
    aliases: Mapping[str, str] | None = None,
    allowed_values: Collection[str] | None = None,
) -> pd.DataFrame:
    """Return a frequency preview of proposed category representations.

    Args:
        values: Scalar category values. Read CSV columns as raw strings when
            original spelling matters; inference may already have changed it.
        case: Case convention applied to the textual representation.
        strip: Remove leading/trailing whitespace. This is a precaution unless
            whitespace variants have actually been observed.
        aliases: Confirmed semantic mappings, applied once after formatting.
            Keys and targets must already follow the selected case/strip rules.
        allowed_values: Authoritative allowed set in the same format. None
            leaves nonblank values unchecked; an empty set allows no values.

    Returns:
        DataFrame with original_value, proposed_value, count and status.
        Original values remain separate even when proposals coincide. Missing
        values stay missing; blank proposals remain blank. Status is missing,
        blank, unchecked, allowed or unexpected. Unexpected values are retained,
        not replaced. Nonmissing proposals use pandas string representations;
        no Boolean or numeric conversion is performed. The input is unchanged.

    Raises:
        ValueError: case is unsupported, or alias/allowed-set entries do not
            follow the chosen formatting convention.
        TypeError: Alias/allowed-set entries are not strings.

    Case conversion is supported by observed variants in this dataset.
    Stripping, retaining unknowns and validating mappings are safeguards, not
    claims that those defects were observed. Aliases encode confirmed meaning,
    not formatting evidence. See the category preview notes under ``docs/``.
    """
    if case not in ("lower", "upper"):
        raise ValueError("case must be 'lower' or 'upper'")

    entries: list[str] = list(allowed_values) if allowed_values is not None else []
    if aliases is not None:
        entries.extend(aliases.keys())
        entries.extend(aliases.values())
    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError("Alias and allowed-set entries must be strings")
        formatted: str = entry.strip() if strip else entry
        formatted = formatted.lower() if case == "lower" else formatted.upper()
        if entry != formatted:
            raise ValueError(
                "Alias and allowed-set entries must use the selected format"
            )

    preview: pd.DataFrame = (
        values.value_counts(dropna=False)
        .rename_axis("original_value")
        .reset_index(name="count")
    )
    proposed: pd.Series = preview["original_value"].astype("string")
    if strip:
        proposed = proposed.str.strip()
    proposed = proposed.str.lower() if case == "lower" else proposed.str.upper()
    if aliases is not None:
        alias_mapping: Mapping[str, str] = aliases
        proposed = proposed.map(
            lambda value: alias_mapping.get(value, value) if pd.notna(value) else value
        ).astype("string")
    preview["proposed_value"] = proposed
    preview["status"] = "unchecked"
    if allowed_values is not None:
        preview["status"] = proposed.isin(allowed_values).map(
            {True: "allowed", False: "unexpected"}
        )
    preview.loc[proposed.eq("").fillna(False), "status"] = "blank"
    preview.loc[proposed.isna(), "status"] = "missing"
    return preview[["original_value", "proposed_value", "count", "status"]]
