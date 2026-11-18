"""Verify category previews preserve evidence and flag unresolved values."""

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from mlops_fraud_prediction_ca1.categories import standardise_categories


class StandardiseCategoriesTests(unittest.TestCase):
    """Exercise formatting, explicit semantics and preservation safeguards."""

    def test_preserves_evidence_and_source(self) -> None:
        """Keep raw groups and distinguish missing, blank and unexpected values."""
        values = pd.Series(["MY", "my", "my", " PK ", "ZZ", "", None])
        original = values.copy(deep=True)
        result = standardise_categories(
            values, case="upper", allowed_values={"MY", "PK"}
        )
        rows = result.set_index("original_value")
        self.assertEqual(rows.loc["my", "count"], 2)
        self.assertEqual(rows.loc["my", "proposed_value"], "MY")
        self.assertEqual(rows.loc[" PK ", "proposed_value"], "PK")
        self.assertEqual(rows.loc["ZZ", "status"], "unexpected")
        self.assertEqual(rows.loc["", "status"], "blank")
        self.assertEqual(
            result.loc[result.original_value.isna(), "status"].item(), "missing"
        )
        self.assertEqual(result["count"].sum(), len(values))
        assert_series_equal(values, original)

    def test_aliases_are_explicit_and_not_chained(self) -> None:
        """Change meaning only via supplied aliases and validate final proposals."""
        values = pd.Series(["YES", "True", "on Delivery"])
        self.assertEqual(standardise_categories(values).iloc[0].proposed_value, "yes")
        result = standardise_categories(
            values,
            aliases={"yes": "true", "true": "confirmed"},
            allowed_values={"true", "confirmed"},
        ).set_index("original_value")
        self.assertEqual(result.loc["YES", "proposed_value"], "true")
        self.assertEqual(result.loc["True", "proposed_value"], "confirmed")
        self.assertEqual(result.loc["on Delivery", "status"], "unexpected")

    def test_optional_trimming_and_empty_allowed_set(self) -> None:
        """Respect disabled stripping and distinguish absent from empty rules."""
        values = pd.Series([" MY "])
        self.assertEqual(
            standardise_categories(values, strip=False).iloc[0].proposed_value, " my "
        )
        self.assertEqual(standardise_categories(values).iloc[0].status, "unchecked")
        self.assertEqual(
            standardise_categories(values, allowed_values=set()).iloc[0].status,
            "unexpected",
        )
        self.assertTrue(standardise_categories(pd.Series([], dtype="string")).empty)

    def test_rejects_inconsistent_configuration(self) -> None:
        """Reject configuration that would silently miss intended matches."""
        values = pd.Series(["MY"])
        with self.assertRaises(ValueError):
            # Deliberately violate the type contract to check runtime validation.
            # noinspection PyTypeChecker
            standardise_categories(values, case="title")
        with self.assertRaises(ValueError):
            standardise_categories(values, aliases={"YES": "true"})
        with self.assertRaises(TypeError):
            # Deliberately supply a non-string category to check runtime validation.
            # noinspection PyTypeChecker
            standardise_categories(values, allowed_values={1})
