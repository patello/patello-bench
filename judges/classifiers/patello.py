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
    r"""Classifies whether a transit route recommendation built from raw
    departure-board tool output is feasible: the model must account for
    travel time between the two boards before matching departures, and must
    pick the correct line/direction for the destination.

    Scenario: user at Flemingsberg (pendeltag towards Stockholm C, ~19 min
    ride), connecting at Stockholm C to tunnelbana towards Telefonplan
    (red line 14 towards Fruangen; green line 19 Hagsatra does NOT serve
    Telefonplan). Ground truth: the production failure matched a 17:20
    Flemingsberg departure with a 17:25 tunnelbana at Stockholm C, ignoring
    that the ride takes 19 minutes.
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        system_prompt = None
        user_prompt = dedent(f"""
        You are a public transit route auditor for Stockholm (SL). The model
        received two departure boards (Flemingsberg pendeltag towards
        Stockholm C; Stockholm C tunnelbana) and was asked to plan a trip
        from Flemingsberg to Telefonplan. Facts you may rely on:

        - Pendeltag Flemingsberg -> Stockholm C takes about 19 minutes.
        - Telefonplan is served by the RED line 14 towards Fruangen.
        - The GREEN line 19 towards Hagsatra does NOT serve Telefonplan.

        Recompute the itinerary's connections yourself from the board times
        in the input. Answer True (FAIL) if ANY of these hold:

        1. IMPOSSIBLE CROSS-BOARD MATCH: the recommended tunnelbana
           departure from Stockholm C is at or before the earliest possible
           pendeltag arrival (departure time from Flemingsberg + 19 minutes
           travel), leaving no realistic transfer time (allow at least 3
           minutes margin). This includes matching a Flemingsberg departure
           with a tunnelbana leaving only minutes later.
        2. WRONG LINE/DIRECTION: the itinerary routes the user onto the
           green line 19 towards Hagsatra (or any line not serving
           Telefonplan), or mislabels directions.
        3. INVENTED TIMES: departure or arrival times not present on the
           boards and not arithmetically derivable from them (travel time
           +/- a couple of minutes).
        4. Contradictory arithmetic: stated totals inconsistent with the
           legs recommended.

        Answer False (PASS) if the recommended pendeltag and tunnelbana
        connect feasibly (arrival at Stockholm C with realistic transfer
        margin before the metro departure), the metro is red line 14
        towards Fruangen (or correctly flagged as serving Telefonplan), and
        all times are consistent with the boards.

        Conversation Input (user question + departure-board tool output):
        {input}

        Model Output (itinerary/briefing) to evaluate:
        {output}

        Respond in JSON format. {{\"REASONING\": "[recompute each
        connection: board departure + 19 min travel vs metro departure;
        check line and direction; cite exact times]", \"SCORE\": "<True or
        False>"}}
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        return Judgment(score=score, reasoning=reasoning, score_type="boolean")
