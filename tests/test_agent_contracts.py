import unittest

from pydantic import ValidationError

from app.agents.analyst_agent import PercentChangeInput, PercentChangeOutput, calculate_percent_change
from app.rag.store import ReportRecord


class AgentContractTests(unittest.TestCase):
    def test_calculation_input_and_output_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            PercentChangeInput(baseline=0, current=100)
        with self.assertRaises(ValidationError):
            PercentChangeInput(baseline=100, current=-1)
        self.assertEqual(calculate_percent_change.invoke({"baseline": 100, "current": 150}), 50.0)
        self.assertEqual(PercentChangeOutput(percent_change=50).percent_change, 50.0)

    def test_report_output_rejects_missing_source(self) -> None:
        with self.assertRaises(ValidationError):
            ReportRecord(source_id="", path="data/report.md", text="evidence")
