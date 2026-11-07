"""
custom_metrics.py - one small DeepEval metric for checks that plain code can decide.

The check is done first (see checks.py), then wrapped in a DeepEval metric so every
result - LLM-judged or not - looks the same in the report and can be gated the same way.
"""

from deepeval.metrics import BaseMetric


class CheckMetric(BaseMetric):
    """A metric whose score was already decided by code (1.0 = pass, 0.0 = fail)."""

    def __init__(self, name, score, reason="", threshold=1.0):
        self._name = name
        self.threshold = threshold
        self.score = score
        self.reason = reason
        self.success = score is not None and score >= threshold
        self.error = None
        self.evaluation_model = "plain code"
        self.include_reason = True
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False

    def measure(self, test_case, *args, **kwargs):
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return self._name
