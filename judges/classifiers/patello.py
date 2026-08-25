"""Patello-bench judges: custom classifiers for real failure modes observed
in production assistants (Swedish-language personal agents).

Failure classes (from reports/patello-bench/private-transcripts/):
1. Invented Swedish words / diacritic corruption / wrong-language intrusions
2. Calorie-budget framing direction (treating remaining time/budget as reassurance)
3. Impossible transit connections (transfer feasibility, direction errors)
"""

from textwrap import dedent

from judges.base import BaseJudge, Judgment


class SwedishFabrication(BaseJudge):
    r"""Classifies whether a Swedish (or Swedish/English mixed) model output
    contains fabricated words, diacritic corruption, or uncontrolled
    wrong-language intrusions.

    Ground-truth examples: "honestare", "tyngder", "kaloridräke",
    "üverskattad" (ü for ö), "crapad", "Bilde"/"underliggende" (Danish/Norwegian
    intrusions into Swedish), "was probably närmare" (mid-sentence switches).
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        system_prompt = None
        user_prompt = dedent(f"""
        You are a strict Swedish language editor. Examine the Model Output for
        fabricated or corrupted language. Answer True if ANY of the following
        are present, otherwise False:

        1. Invented words: words that do not exist in standard Swedish
           (SAOL/SO) or standard English, even if they look plausible.
           Examples of invented words: "honestare", "tyngder" (correct:
           "tungare"), "kaloridräke", "crapad", "övertrassling",
           "jagimiterade", "föändra", "sänxt".
        2. Diacritic corruption: wrong diacritics in Swedish words, e.g.
           "üverskattad" (correct: "överskattad"), "übert".
        3. Wrong-language intrusions: Danish/Norwegian/German words used
           where Swedish is required, e.g. "Bilde" (Swedish: "bild"),
           "underliggende" (Swedish: "underliggande"), "schnitt".
        4. Uncontrolled code-switching: English fragments inside Swedish
           sentences that a fluent writer would not produce, e.g.
           "I körde på", "Wants du that jag...", "So: 300 idag är probably
           the honestare one."

        Do NOT flag: correct Swedish, correct English, correct proper nouns,
        place names, technical loanwords in normal use, or intentional
        light-code-switching that is grammatical in both languages.

        Conversation Input (for context):
        {input}

        Model Output to evaluate:
        {output}

        Respond in JSON format. {{\"REASONING\": \"[list each flagged word and
        its class, or state none found]\", \"SCORE\": \"<True or False>\"}}
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        return Judgment(score=score, reasoning=reasoning, score_type="boolean")


class KcalBudgetFraming(BaseJudge):
    r"""Classifies whether a calorie-coach comment frames the daily budget in
    the wrong direction.

    The goal is to come in UNDER a daily kcal target. Treating remaining
    time or remaining budget as reassurance ("it's only 2pm, you have all
    evening") is backwards: more day means more eating ahead.
    Ground truth: the 2026-07-20 "allt är bakvänt" episode.
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        system_prompt = None
        user_prompt = dedent(f"""
        You are evaluating a calorie-coaching assistant's comment. The user's
        goal is to come in UNDER a daily calorie target (e.g. 1800 kcal).
        Judge whether the comment frames progress in the correct direction.

        Answer True (FAIL) if the comment does any of these:
        - Frames remaining time of day as reassurance when the total is high
          for that time ("it's only 2pm" / "du har hela kvällen kvar" used to
          dismiss concern — more day means more eating ahead).
        - Frames remaining budget as something to use up ("you still have
          600 kcal to spend") when the standing instructions say headroom is
          a buffer, not a budget.
        - Dismisses the user's concern about a high total by pointing at
          remaining budget or time.
        - Encourages eating to reach the target.
        - FALSE REASSURANCE OF MARGIN: first compute the real margin
          (target minus logged minus planned). If the real margin is zero or
          negative and the output claims there is room to spare ("plenty of
          room", "god marginal", "no problem"), that is a failure (e.g.
          logged 1600 + dinner 300 + dessert 300 = 2200 vs target 1800).
          Do NOT fail an output merely for calling a genuinely positive
          remaining margin "solid" or "good" — a small but real buffer
          accurately stated is a pass, even if the phrasing is optimistic.
        - Ignores or contradicts explicit standing instructions in the
          conversation about staying under the target.

        Answer False (PASS) if the comment:
        - Celebrates being under target ("good margin to goal", "solid day")
          when the arithmetic actually supports it.
        - Uses time of day correctly: reassures ONLY when the total is
          genuinely low for that time (e.g. 400 kcal at 15:00).
        - Acknowledges concern about high totals honestly, states the
          correct remaining buffer (which may be zero or negative).
        - Reports numbers neutrally without spin.

        Conversation Input (logged meals, totals, time of day, user concern):
        {input}

        Model Output to evaluate:
        {output}

        Respond in JSON format. {{\"REASONING\": \"[explain the framing
        direction and cite the exact phrase]\", \"SCORE\": \"<True or False>\"}}
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        return Judgment(score=score, reasoning=reasoning, score_type="boolean")


class TransitConnectionFeasibility(BaseJudge):
    r"""Classifies whether a transit (SL Stockholm) route recommendation is
    feasible: connections must allow a realistic transfer (>= 5 minutes),
    connect in the correct order (no departures before arrival), and use
    correct direction labels.

    Ground truth: 2026-05-19 briefing suggesting a bus 1 minute after train
    arrival, labeled "tight" but presented as primary option; "nordgående"
    (northbound) used for southbound trains.
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        system_prompt = None
        user_prompt = dedent(f"""
        You are a public transit route auditor for Stockholm (SL). Examine
        the recommended itinerary. Answer True (FAIL) if ANY of these hold:

        1. A connecting departure is scheduled at or before the arrival of
           the leg it connects from (negative or <5 min transfer time),
           unless the connection is clearly marked as not recommended and a
           realistic alternative is presented first.
        2. A physically impossible sequence (departure before the passenger
           can possibly arrive).
        3. Direction labels are wrong (e.g. calling trains toward
           Södertälje/Tumba "northbound"; those are southbound from
           Stockholm).
        4. Total journey time inconsistent with the legs given.

        Answer False (PASS) if all connections allow >= 5 minutes transfer,
        sequence is possible, directions are correct, and times add up.

        Conversation Input (user location, departure board data):
        {input}

        Model Output (itinerary/briefing) to evaluate:
        {output}

        Respond in JSON format. {{\"REASONING\": \"[check each connection:
        arrival vs departure, transfer minutes, direction; cite exact
        times]\", \"SCORE\": \"<True or False>\"}}
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        return Judgment(score=score, reasoning=reasoning, score_type="boolean")
