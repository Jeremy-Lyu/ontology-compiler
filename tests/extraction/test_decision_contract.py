"""Offline decision contract: boolean / choice / score, no torch or LLM."""
from __future__ import annotations

import math
import unittest

from ontology_compiler.decisions import (
    answer_from_probabilities,
    candidate_ids,
    decision_score,
    expected_calibration_error,
    multiclass_nll,
    scalar_brier,
    validate_request,
)
from ontology_compiler.decisions.teacher import TeacherError, parse_hard_labels


def _payload():
    return {
        "states": [
            {
                "id": "doc_001",
                "state": {"chunk": "MBOM 缺料停线。"},
                "questions": {
                    "dup": {"type": "boolean", "instructions": "是否重复？"},
                    "domain": {
                        "type": "choice",
                        "instructions": "领域？",
                        "criteria": {"A": "制造。", "B": "质量。"},
                    },
                    "q": {
                        "type": "score",
                        "instructions": "质量？",
                        "criteria": ["差。", "中。", "好。"],
                    },
                },
            }
        ]
    }


class DecisionContractTest(unittest.TestCase):
    def test_validate_ok(self):
        parsed = validate_request(_payload())
        self.assertEqual(len(parsed.states), 1)

    def test_validate_rejects_extra_key(self):
        with self.assertRaises(ValueError):
            validate_request({"states": [], "teacher": {}})

    def test_validate_rejects_bad_choice(self):
        bad = _payload()
        bad["states"][0]["questions"]["domain"]["criteria"] = {"only": "一个候选"}
        with self.assertRaises(ValueError):
            validate_request(bad)

    def test_answer_math(self):
        parsed = validate_request(_payload())
        qs = parsed.states[0].questions
        boolean = answer_from_probabilities(qs["dup"], [0.3, 0.7])
        self.assertEqual(boolean["p_true"], 0.7)
        self.assertIs(boolean["value"], True)
        choice = answer_from_probabilities(qs["domain"], [0.2, 0.8])
        self.assertEqual(choice["choice"], "B")
        score = answer_from_probabilities(qs["q"], [0.1, 0.2, 0.7])
        self.assertEqual(score["level"], 2)
        self.assertAlmostEqual(score["score"], 0 * 0.1 + 1 * 0.2 + 2 * 0.7)
        self.assertEqual(candidate_ids(qs["dup"]), ["false", "true"])
        self.assertAlmostEqual(decision_score([0.1, 0.2, 0.7]), score["score"])

    def test_answer_rejects_bad_sum(self):
        parsed = validate_request(_payload())
        with self.assertRaises(ValueError):
            answer_from_probabilities(parsed.states[0].questions["dup"], [0.5, 0.6])

    def test_metrics(self):
        self.assertAlmostEqual(scalar_brier(0.7, True), 0.09)
        self.assertAlmostEqual(multiclass_nll([0.2, 0.8], 1), -math.log(0.8))
        ece = expected_calibration_error([0.9, 0.1], [True, False], n_bins=2)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_teacher_strict(self):
        questions = _payload()["states"][0]["questions"]
        ok = parse_hard_labels('{"answers":{"dup":true,"domain":"A","q":2}}', questions)
        self.assertEqual(ok, {"dup": True, "domain": "A", "q": 2})
        rejected = [
            '{"answers":{"dup":1,"domain":"A","q":2}}',
            '{"answers":{"dup":true,"domain":"ZZZ","q":2}}',
            '{"answers":{"dup":true,"domain":"A","q":5}}',
            '{"answers":{"dup":true,"domain":"A"}}',
            '```json {"answers":{"dup":true,"domain":"A","q":2}} ```',
        ]
        for bad in rejected:
            with self.subTest(bad=bad):
                with self.assertRaises(TeacherError):
                    parse_hard_labels(bad, questions)


if __name__ == "__main__":
    unittest.main()
