r"""Regression tests for match(numeric=True) when the final answer is wrapped.

A math-tuned model frequently ends with `ANSWER: \boxed{N}`. Before the fix,
the location="end" scan could not parse the wrapper, silently stepped past it
into the reasoning, and graded whichever number appeared last there. Both
directions were wrong: a wrong final answer was credited off the working, and a
correct final answer was rejected in favour of a number from the working.
"""

import pytest
from test_helpers.utils import simple_task_state

from inspect_ai.scorer import CORRECT, INCORRECT, Target, match


@pytest.mark.anyio
async def test_boxed_final_wrong_answer_not_credited_from_reasoning():
    # the reasoning ends in 18, the declared final answer is 126
    scorer = match(numeric=True)
    state = simple_task_state(
        model_output="She earns 9 * 2 = 18 dollars a day.\n\nANSWER: \\boxed{126}"
    )
    result = await scorer(state, Target(["18"]))

    assert result.text == INCORRECT
    assert result.answer == "126"


@pytest.mark.anyio
async def test_boxed_final_correct_answer_not_rejected_for_reasoning():
    # the reasoning ends in 2, the declared final answer is 18
    scorer = match(numeric=True)
    state = simple_task_state(
        model_output="Janet sells them at $2 per egg.\n\nANSWER: \\boxed{18}"
    )
    result = await scorer(state, Target(["18"]))

    assert result.text == CORRECT
    assert result.answer == "18"


@pytest.mark.anyio
async def test_plain_final_answer_control():
    # a parseable final answer was already graded correctly; it must stay so
    scorer = match(numeric=True)
    state = simple_task_state(model_output="ANSWER: 18")
    result = await scorer(state, Target(["18"]))

    assert result.text == CORRECT
    assert result.answer == "18"


@pytest.mark.anyio
async def test_boxed_final_answer_wrong_is_incorrect():
    scorer = match(numeric=True)
    state = simple_task_state(
        model_output="Janet sells them at $2 per egg.\n\nANSWER: \\boxed{19}"
    )
    result = await scorer(state, Target(["18"]))

    assert result.text == INCORRECT
    assert result.answer == "19"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "model_output",
    [
        "first 7 then 25 then done\nANSWER: \\boxed{25}",
        "The total is $1,234.00.\n\nAnswer: \\boxed{1234}",
        "compute 3 + 4\n\\boxed{7}",
        "step one gives 9\nFINAL ANSWER: \\boxed{10.5}",
        "she has 5 apples\nANSWER: \\boxed{(42)}",
    ],
)
async def test_boxed_wrappers_are_unwrapped_before_parsing(model_output: str):
    # the number inside the wrapper is the answer, whatever the reasoning said
    scorer = match(numeric=True)
    state = simple_task_state(model_output=model_output)
    expected = {
        "first 7 then 25 then done\nANSWER: \\boxed{25}": "25",
        "The total is $1,234.00.\n\nAnswer: \\boxed{1234}": "1234",
        "compute 3 + 4\n\\boxed{7}": "7",
        "step one gives 9\nFINAL ANSWER: \\boxed{10.5}": "10.5",
        "she has 5 apples\nANSWER: \\boxed{(42)}": "42",
    }[model_output]

    result = await scorer(state, Target([expected]))
    assert result.text == CORRECT
    assert result.answer == expected


@pytest.mark.anyio
async def test_unparseable_final_after_marker_does_not_fall_through():
    # the marker announces the answer, the answer is not a number, so nothing
    # from the working may be graded in its place
    scorer = match(numeric=True)
    state = simple_task_state(
        model_output="Janet sells them at $2 per egg.\n\nANSWER: \\boxed{two}"
    )
    result = await scorer(state, Target(["2"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_uppercase_marker_bounds_the_scan_when_case_sensitive():
    # marker detection must not depend on ignore_case: an unparseable "ANSWER:"
    # has to bound the scan for a case-sensitive caller too, otherwise the
    # reasoning number is graded in its place
    scorer = match(numeric=True, ignore_case=False)
    state = simple_task_state(model_output="Step: 18 apples.\nANSWER: unknown")
    result = await scorer(state, Target(["18"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_answer_before_marker_still_graded():
    # "42 is the answer" carries no marker boundary, so the trailing words must
    # not suppress the number that precedes them
    scorer = match(numeric=True)
    state = simple_task_state(model_output="42 is the answer")
    result = await scorer(state, Target(["42"]))

    assert result.text == CORRECT
    assert result.answer == "42"


@pytest.mark.anyio
async def test_boxed_nested_wrapper():
    scorer = match(numeric=True)
    state = simple_task_state(model_output="work: 3\n\nANSWER: \\boxed{\\boxed{11}}")
    result = await scorer(state, Target(["11"]))

    assert result.text == CORRECT
    assert result.answer == "11"


@pytest.mark.anyio
async def test_boxed_with_trailing_period():
    scorer = match(numeric=True)
    state = simple_task_state(model_output="work: 3\n\nANSWER: \\boxed{11}.")
    result = await scorer(state, Target(["11"]))

    assert result.text == CORRECT
    assert result.answer == "11"


@pytest.mark.anyio
async def test_markdown_bold_marker_and_boxed():
    scorer = match(numeric=True)
    state = simple_task_state(model_output="work: 3\n\n**ANSWER: \\boxed{11}**")
    result = await scorer(state, Target(["11"]))

    assert result.text == CORRECT
    assert result.answer == "11"


@pytest.mark.anyio
async def test_boxed_negative_and_decimal():
    scorer = match(numeric=True)
    state = simple_task_state(model_output="gain of 4\nANSWER: \\boxed{-3.5}")
    result = await scorer(state, Target(["-3.5"]))

    assert result.text == CORRECT
    assert result.answer == "-3.5"


@pytest.mark.anyio
async def test_boxed_answer_location_begin():
    # location="begin" means "the first number in the value", so a leading
    # wrapper is unwrapped there too
    scorer = match(numeric=True, location="begin")
    state = simple_task_state(model_output="\\boxed{11} is my answer, 4 was wrong")
    result = await scorer(state, Target(["11"]))

    assert result.text == CORRECT
    assert result.answer == "11"


@pytest.mark.anyio
async def test_boxed_answer_location_any():
    scorer = match(numeric=True, location="any")
    state = simple_task_state(model_output="first \\boxed{7} then 25")
    result = await scorer(state, Target(["7"]))

    assert result.text == CORRECT
    assert result.answer == "7"


@pytest.mark.anyio
async def test_boxed_answer_location_exact_stays_exact():
    # exact must not gain wrapper trimming
    scorer = match(numeric=True, location="exact")
    state = simple_task_state(model_output="\\boxed{42}")
    result = await scorer(state, Target(["42"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_non_numeric_match_unaffected_by_boxed():
    # numeric=False keeps its plain text comparison and does not unwrap, so a
    # boxed output only matches a boxed target
    scorer = match()
    state = simple_task_state(model_output="ANSWER: \\boxed{18}")

    assert (await scorer(state, Target(["18"]))).text == CORRECT
    assert (await scorer(state, Target(["boxed{18}"]))).text == CORRECT
