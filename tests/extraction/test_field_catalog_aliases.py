"""Field catalog alias loading for structured Excel columns."""
from __future__ import annotations

import unittest

import pandas as pd

from ontology_compiler.services.extraction.structured_data_extractor import (
    _load_catalog_aliases,
    normalize_columns,
)


class FieldCatalogAliasTest(unittest.TestCase):
    def test_supply_chain_uses_inventory_catalog_and_synonyms(self):
        aliases = _load_catalog_aliases("SUPPLY_CHAIN")
        self.assertEqual(
            aliases.get("3 months avg daily demand"),
            "3_months_avg_daily_demand",
        )
        # Catalog alias wins over the shorter governance synonym ("pn").
        self.assertEqual(aliases.get("part number"), "part_number")
        self.assertEqual(aliases.get("article"), "pn")
        self.assertEqual(aliases.get("safety stock"), "safety_stock")

    def test_manufacturing_shares_the_inventory_catalog(self):
        inventory = _load_catalog_aliases("inventory")
        manufacturing = _load_catalog_aliases("manufacturing")
        self.assertEqual(
            manufacturing.get("3 months avg daily demand"),
            inventory.get("3 months avg daily demand"),
        )

    def test_unknown_domain_still_gets_synonym_baseline(self):
        aliases = _load_catalog_aliases("warranty")
        self.assertEqual(aliases.get("lead time"), "lead_time")
        self.assertNotIn("3 months avg daily demand", aliases)

    def test_normalize_columns_renames_known_headers(self):
        frame = pd.DataFrame(columns=["Part Number", "Safety Stock", "Article", "Notes"])
        renamed = normalize_columns(frame, domain="SUPPLY_CHAIN")
        self.assertEqual(list(renamed.columns), ["part_number", "safety_stock", "pn", "Notes"])


if __name__ == "__main__":
    unittest.main()
