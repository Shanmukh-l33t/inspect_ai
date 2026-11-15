import math
import re
from typing import Callable, Literal

from inspect_ai._util.text import (
    strip_numeric_punctuation,
    strip_punctuation,
)
from inspect_ai.scorer._unicode import unicode_number_to_float
from inspect_ai.solver._task_state import TaskState

from ._metric import CORRECT, INCORRECT, Score
from ._scorer import Scorer
from ._target import Target


def no_response(completion: str) -> bool:
    """Whether the model produced nothing that could be scored.

    An empty or whitespace-only completion never produced output that could
    violate a format, so `no_response` is the correct `ScoreReason` for it
    rather than `invalid_response_format`.

    Args:
        completion: The raw model completion, never an extracted answer.

    Returns:
        True when the raw completion is empty or whitespace only.
    """
    return not completion.strip()


def str_match_scorer(match: Callable[[str, str], tuple[str, bool]]) -> Scorer:
    """Scorer that uses a matching function.

    The matching function returns tuple[str,bool], where str is the answer
    extracted from the model output and bool is whether it matched the target
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Decided once, above the loop, because a raw-empty completion can reach
        # the CORRECT return as well as the INCORRECT one. match() with a target
        # of "." normalizes that target to empty, and "".endswith("") is True;
        # includes() with an empty target is contained in anything. Both score
        # CORRECT off a completion the model never produced. The accepted scope
        # tags a raw-empty completion regardless of the value it ended up with,
        # so the value is preserved here and only the reason is added.
        empty = no_response(state.output.completion or "")
        answer: str | None = None
        for value in target:
            answer, matched = match(state.output.completion, value)
            if matched:
                return Score(
                    value=CORRECT,
                    answer=answer,
                    reason="no_response" if empty else None,
                    explanation=state.output.completion,
                )

        return Score(
            value=INCORRECT,
            answer=answer,
            reason="no_response" if empty else None,
            explanation=state.output.completion,
        )

    return score


def match_str(
    value: str,
    target: str,
    location: Literal["begin", "end", "any", "exact"] = "end",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    numeric: bool = False,
) -> tuple[str, bool]:
    # strip ws
    v = value.strip()
    t = target.strip()

    # baseline answer (will only change for numeric)
    answer = v

    # further cleanup
    if ignore_case:
        v = v.casefold()
        t = t.casefold()
    if numeric:
        t = strip_numeric_punctuation(t)
    if numeric and _is_number(t):
        # the target is a number: extract the relevant number(s) from the
        # value, normalize both, and compare for numeric equality. we do
        # NOT fall through to the text comparison below because e.g.
        # "25".endswith("5") is True but 25 != 5.
        v = strip_numeric_punctuation(v)
        t = normalize_number(t)
        if location != "exact":
            # a math-tuned model frequently wraps its final answer, e.g.
            # "ANSWER: \boxed{126}". Left wrapped, the token does not parse
            # and the end-of-value scan below would step past it into the
            # reasoning and grade a number from the working. exact is
            # excluded: it stays actually exact, wrappers included.
            v = _unwrap_answer_wrappers(v)
        words = re.split(r"\s+", v)
        if location == "begin":
            v = first_number_normalized(words)
        elif location == "end":
            # scan only the text after the last answer marker ("Answer: 42",
            # "the result is 42") when there is one, so an announced final
            # answer that does not parse scores as no number rather than
            # falling through into the reasoning that precedes it
            markers = list(_ANSWER_MARKER.finditer(v))
            if markers:
                words = re.split(r"\s+", v[markers[-1].end() :])
            words.reverse()
            v = first_number_normalized(words)
        elif location == "exact":
            # exact stays actually exact: no punctuation trimming, so
            # "(42)" does not match a target of "42" under exact
            v = normalize_number(v, trim_punctuation=False)
        else:
            # location == "any": match if any number in the value equals t
            for number in all_numbers_normalized(words):
                if number == t:
                    return number, True
            return answer, False
        answer = v
        return answer, v == t
    elif ignore_punctuation:
        v = strip_punctuation(v)
        t = strip_punctuation(t)

    # comparisons
    if location == "begin":
        return answer, v.startswith(t)
    elif location == "end":
        return answer, v.endswith(t)
    elif location == "exact":
        return answer, v == t
    else:
        return answer, t in v


# Sentence/enclosing punctuation only, including the full-width forms of
# those characters, the ideographic full stop ("42。", "（42）"), and the CJK
# corner brackets used as quotes ("「42」", "『42』", "【42】"). Operator
# characters (~ < > = etc.) are deliberately excluded: "<42", ">=42", "~42"
# express bounds or approximations, not the value itself, and trimming them
# would grade a hedge as the exact answer (false CORRECTs). U+FF0E
# FULLWIDTH FULL STOP is excluded too: like ASCII ".", it is a decimal
# separator ("．42" is 0.42).
_NUMERIC_PUNCTUATION_TRIM = "!?:;()[]{}'\"`！？：；（）［］｛｝＇＂｀。「」『』【】"

# Same marker vocabulary as the math() scorer (_math.py), so both scorers
# announce a final answer the same way. The trailing "is\b|[:=]" keeps it
# from firing on ordinary prose ("the answer is not known") and on
# arithmetic ("9 * 2 = 18"), which is working, not an announced answer. No
# IGNORECASE flag: the value is already casefolded when ignore_case is on,
# and a caller who turned case folding off should not get it back here.
_ANSWER_MARKER = re.compile(r"(?:final\s+answer|answer|result)\s*(?:is\b|[:=])\s*")

# Final answers wrapped in a LaTeX box, e.g. "\boxed{18}", "\fbox{18}", or a
# bare "boxed{18}". Mirrors _BOX_START in _math.py.
_BOX_START = re.compile(
    r"(?:\\(?:beginboxed|boxed|fbox)|(?<![A-Za-z\\])(?:boxed|fbox|oxed))\s*\{"
)


def _unwrap_answer_wrappers(value: str, max_depth: int = 8) -> str:
    r"""Replace each answer wrapper with the text it encloses.

    "\boxed{18}" becomes "18" and "\boxed{\boxed{18}}" likewise, so the
    numeric scan sees the answer itself instead of an unparseable token.
    An unterminated wrapper is left as is: there is no answer inside it to
    recover, and the rest of the value is still scanned normally.
    """
    for _ in range(max_depth):
        box = _BOX_START.search(value)
        if box is None:
            break
        close = _find_matching_brace(value, box.end() - 1)
        if close is None:
            break
        value = value[: box.start()] + value[box.end() : close] + value[close + 1 :]
    return value


def _find_matching_brace(value: str, open_index: int) -> int | None:
    """Index of the brace closing the one at `open_index`, or None."""
    depth = 0
    for index in range(open_index, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _clean_numeric_word(s: str) -> str:
    return s.strip(_NUMERIC_PUNCTUATION_TRIM)


def _parse_number(s: str, *, trim_punctuation: bool = True) -> float | None:
    """Parse `s` as a finite number, or None.

    Recognises plain float syntax (signs, decimals, exponents) and the
    full set of inputs handled by ``unicode_number_to_float`` (unicode
    minus, vulgar fractions, superscripts, Chinese numerals, fullwidth
    digits, locale grouping). Rejects strings with trailing non-numeric
    content (so ``"5 some text"`` returns None). Sentence/enclosing
    punctuation attached to the number ("42!", "(42)", "42。") is trimmed unless
    ``trim_punctuation=False`` — the ``location="exact"`` path passes False
    so that exact stays actually exact. ``nan`` and ``inf`` also return None.
    """
    s_cleaned = _clean_numeric_word(s) if trim_punctuation else s
    for parse in (float, unicode_number_to_float):
        try:
            num = parse(s_cleaned)
        except ValueError:
            continue
        if math.isfinite(num):
            return num
    return None


def _is_number(s: str) -> bool:
    return _parse_number(s) is not None


def first_number_normalized(words: list[str]) -> str:
    number = next((word for word in words if _is_number(word)), None)
    return normalize_number(number) if number is not None else ""


def all_numbers_normalized(words: list[str]) -> list[str]:
    return [normalize_number(word) for word in words if _is_number(word)]


def normalize_number(
    number: str, precision: int = 5, *, trim_punctuation: bool = True
) -> str:
    num = _parse_number(number, trim_punctuation=trim_punctuation)
    if num is None:
        return number
    return format(num, f".{precision}g")
