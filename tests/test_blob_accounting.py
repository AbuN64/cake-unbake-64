import unittest

from unbake import blobs, config


class AccountingTests(unittest.TestCase):
    def test_observed_rows_account_for_bytes_without_credit(self):
        p = blobs.sweep(256, [(0, 32)], [], [], [], [])
        document = {"blob": {"a": {"state": "observed", "rom": {"v": [[32, 128]]}}}}
        self.assertEqual(blobs.no_row(p, document, "v"), 128)
        self.assertEqual(p["blobs_decoded"], 0)
        self.assertEqual(p["unknown"], 224)

    def test_overlap_and_children_cannot_double_count(self):
        p = blobs.sweep(256, [(0, 32)], [(32, 64)], [(64, 96)], [], [])
        document = {"blob": {
            "a": {"state": "observed", "rom": {"v": [[32, 160], [150, 256]]}},
            "child": {"parent": "a", "decoded": [0, 200], "state": "decoded"},
        }}
        self.assertEqual(blobs.no_row(p, document, "v"), 0)
        self.assertEqual(blobs.no_row(p, document, "other"), 160)

    def test_accounting_kinds_validate_without_semantic_promotion(self):
        for kind in ("sn-debug", "stale-code", "stale-data", "fill", "metadata", "unclassified"):
            doc = {"schema": 1, "blob": {"a": {
                "type": kind, "format": "explicit accounting observation",
                "state": "observed", "rom": {"v": [[32, 64]]},
            }}}
            config.validate("blobs", doc, "blobs.toml")
            self.assertEqual(blobs.ranges(doc, "v"), ([], []))


if __name__ == "__main__":
    unittest.main()
