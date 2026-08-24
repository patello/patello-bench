from unittest.mock import patch, MagicMock

from judges.base import Judgment
from judges.classifiers.patello import (
    SwedishFabrication,
    KcalBudgetFraming,
    TransitConnectionFeasibility,
)

import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def mock_judgment(judgment: Judgment):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with patch("instructor.from_provider") as mock_from_provider:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = judgment
                mock_from_provider.return_value = mock_client
                return func(self, mock_client.chat.completions.create, *args, **kwargs)
        return wrapper
    return decorator


class TestSwedishFabrication:

    @mock_judgment(Judgment(score=True, reasoning="flagged: honestare, üverskattad", score_type="boolean"))
    def test_flags_fabrication(self, mock_create):
        j = SwedishFabrication(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(
            input="scones vs hönökaka discussion",
            output="300 idag är probably the honestare one. Igår's 450 var nog üverskattad.",
        )
        assert judgment.score is True
        mock_create.assert_called_once()

    @mock_judgment(Judgment(score=False, reasoning="clean Swedish", score_type="boolean"))
    def test_passes_clean(self, mock_create):
        j = SwedishFabrication(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(input="log breakfast", output="+300, 300/1800")
        assert judgment.score is False


class TestKcalBudgetFraming:

    @mock_judgment(Judgment(score=True, reasoning="remaining time framed as reassurance", score_type="boolean"))
    def test_flags_backwards_framing(self, mock_create):
        j = KcalBudgetFraming(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(
            input="970 kcal at 14:00, target under 1800",
            output="Klockan är bara halv tre — du har hela kvällen kvar. Ingen fara alls.",
        )
        assert judgment.score is True

    @mock_judgment(Judgment(score=False, reasoning="correct direction", score_type="boolean"))
    def test_passes_good_framing(self, mock_create):
        j = KcalBudgetFraming(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(
            input="400 kcal at 15:00, target under 1800",
            output="Bara 400 vid halv fyra — under din vanliga takt. Bra marginal.",
        )
        assert judgment.score is False


class TestTransitConnectionFeasibility:

    @mock_judgment(Judgment(score=True, reasoning="bus departs before train arrives", score_type="boolean"))
    def test_flags_impossible_connection(self, mock_create):
        j = TransitConnectionFeasibility(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(
            input="train arrives Södra 16:25",
            output="Ta tåget 16:13, ankomst 16:25 — anslut buss 66 kl 16:14 (1 min marginal).",
        )
        assert judgment.score is True

    @mock_judgment(Judgment(score=False, reasoning="14 min transfer, feasible", score_type="boolean"))
    def test_passes_feasible_connection(self, mock_create):
        j = TransitConnectionFeasibility(model="openrouter/google/gemini-2.5-flash")
        judgment = j.judge(
            input="bus arrives Södra 07:13",
            output="Buss 66 kl 06:59, ankomst 07:13, byt till tåg 41 kl 07:27 — 14 minuters byte.",
        )
        assert judgment.score is False
