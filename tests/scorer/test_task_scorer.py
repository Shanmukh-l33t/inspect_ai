import pytest

from inspect_ai import eval, score
from inspect_ai._eval.task.task import Task
from inspect_ai._util.registry import (
    registry_info,
    registry_params,
    registry_unqualified_name,
)
from inspect_ai.dataset import Sample
from inspect_ai.dataset._sources.csv import csv_dataset
from inspect_ai.scorer import includes, stderr
from inspect_ai.scorer._answer import answer
from inspect_ai.scorer._classification import f1
from inspect_ai.scorer._metrics import accuracy, mean
from inspect_ai.scorer._metrics.std import bootstrap_stderr
from inspect_ai.scorer._scorer import scorer_metrics


def test_task_with_metrics():
    task = Task(scorer=f1(), metrics=[mean(), bootstrap_stderr()])

    # ensure that metrics themselves remain unchanged
    assert registry_info(task.metrics[0]).name == "inspect_ai/mean"
    assert registry_info(task.metrics[1]).name == "inspect_ai/bootstrap_stderr"
    assert task.scorer is not None

    # ensure that the task metrics are correctly applied to the scorer
    info = registry_info(task.scorer[0])
    assert registry_info(info.metadata["metrics"][0]).name == "inspect_ai/mean"

    info = registry_info(task.scorer[0])
    assert (
        registry_info(info.metadata["metrics"][1]).name == "inspect_ai/bootstrap_stderr"
    )

    # modify the task and ensure that the new metrics remain unchanged
    task.scorer.append(answer("word"))
    assert len(task.scorer) == 2
    info = registry_info(task.scorer[1])
    assert registry_info(info.metadata["metrics"][0]).name == "inspect_ai/accuracy"
    assert registry_info(info.metadata["metrics"][1]).name == "inspect_ai/stderr"


def test_task_score_results():
    task = Task(
        dataset=csv_dataset("tests/dataset/test_dataset/samples-md.csv"),
        scorer=f1(),
        metrics=[accuracy()],
    )

    # confirm the mean result is computed
    log = eval(task, model="mockllm/model")
    assert len(log[0].results.scores) == 1
    assert len(log[0].results.scores[0].metrics) == 1
    assert "accuracy" in log[0].results.scores[0].metrics
    assert "mean" not in log[0].results.scores[0].metrics
    assert "stderr" not in log[0].results.scores[0].metrics


def test_score_results():
    task = Task(
        dataset=csv_dataset("tests/dataset/test_dataset/samples-md.csv"),
        scorer=f1(),
    )

    # confirm the mean result is computed
    log = eval(task, model="mockllm/model")
    assert len(log[0].results.scores) == 1
    assert len(log[0].results.scores[0].metrics) == 2
    assert "mean" in log[0].results.scores[0].metrics
    assert "stderr" in log[0].results.scores[0].metrics


def test_added_scores():
    task = Task(
        dataset=csv_dataset("tests/dataset/test_dataset/samples-md.csv"),
        scorer=f1(),
        metrics=[accuracy()],
    )
    task.scorer.append(answer("line"))

    log = eval(task, model="mockllm/model")
    assert len(log[0].results.scores) == 2
    assert len(log[0].results.scores[0].metrics) == 1
    assert "accuracy" in log[0].results.scores[0].metrics
    assert "mean" not in log[0].results.scores[0].metrics
    assert "stderr" not in log[0].results.scores[0].metrics

    assert len(log[0].results.scores[1].metrics) == 2
    assert "accuracy" in log[0].results.scores[1].metrics
    assert "stderr" in log[0].results.scores[1].metrics


# metrics that the second task and the grader in these tests ask for
PLAIN_STDERR = 0.1889822365046136
CLUSTERED_STDERR = 0.28867513459481287


def _passage_dataset() -> list[Sample]:
    # mockllm answers "Default output from mockllm/model": samples 1, 2, 5 and 6
    # are correct. Consecutive samples share a passage so they can be clustered.
    return [
        Sample(
            input="q",
            target="Default" if i in (1, 2, 5, 6) else "absent",
            id=i,
            metadata={"passage": (i - 1) // 2},
        )
        for i in range(1, 9)
    ]


def _clustered_metrics():
    return [accuracy(), stderr(cluster="passage")]


def _stderr_metric(log):
    assert log.results is not None
    return log.results.scores[0].metrics["stderr"]


def test_task_metrics_do_not_change_a_shared_scorer():
    # one scorer object used by two tasks, only the second asks for clustered
    # standard errors
    shared = includes()
    overall = Task(dataset=_passage_dataset(), scorer=shared, name="overall")
    by_passage = Task(
        dataset=_passage_dataset(),
        scorer=shared,
        metrics=_clustered_metrics(),
        name="by_passage",
    )

    # the task carries a scorer of its own, named and parameterised like the
    # one it was given, so its log records the same scorer spec
    assert by_passage.scorer is not None
    assert by_passage.scorer[0] is not shared
    assert registry_unqualified_name(by_passage.scorer[0]) == "includes"
    assert registry_params(by_passage.scorer[0]) == {}

    # creating the second task leaves the shared scorer's own metrics alone
    shared_metrics = scorer_metrics(shared)
    assert [registry_unqualified_name(metric) for metric in shared_metrics] == [
        "accuracy",
        "stderr",
    ]
    assert registry_params(shared_metrics[1]) == {}

    # so the first task reports its own stderr, not the clustered one
    overall_log = eval(overall, model="mockllm/model", display="none")[0]
    stderr_metric = _stderr_metric(overall_log)
    assert stderr_metric.value == pytest.approx(PLAIN_STDERR)
    assert stderr_metric.params == {}

    # and the task that asked for clustered stderr still gets it
    by_passage_log = eval(by_passage, model="mockllm/model", display="none")[0]
    stderr_metric = _stderr_metric(by_passage_log)
    assert stderr_metric.value == pytest.approx(CLUSTERED_STDERR)
    assert stderr_metric.params == {"cluster": "passage"}


def test_score_metrics_do_not_change_a_shared_grader():
    # x records task metrics, y does not
    x, y = eval(
        [
            Task(
                dataset=_passage_dataset(),
                scorer=includes(),
                metrics=_clustered_metrics(),
                name="x",
            ),
            Task(dataset=_passage_dataset(), scorer=includes(), name="y"),
        ],
        model="mockllm/model",
        display="none",
    )

    # one grader rescoring both logs uses each log's own metrics
    grader = includes()
    stderr_metric = _stderr_metric(score(x, grader, action="overwrite", display="none"))
    assert stderr_metric.value == pytest.approx(CLUSTERED_STDERR)
    assert stderr_metric.params == {"cluster": "passage"}

    stderr_metric = _stderr_metric(score(y, grader, action="overwrite", display="none"))
    assert stderr_metric.value == pytest.approx(PLAIN_STDERR)
    assert stderr_metric.params == {}


def test_appended_scorer_uses_its_own_metrics_after_task_metrics():
    # a grader used with task metrics must still append with its own metrics
    clustered_log = eval(
        Task(
            dataset=_passage_dataset(),
            scorer=includes(),
            metrics=_clustered_metrics(),
        ),
        model="mockllm/model",
        display="none",
    )[0]
    plain_log = eval(
        Task(dataset=_passage_dataset(), scorer=includes()),
        model="mockllm/model",
        display="none",
    )[0]

    grader = includes()
    score(clustered_log, grader, action="overwrite", display="none")

    appended_log = score(plain_log, grader, action="append", display="none")
    assert appended_log.results is not None
    appended = appended_log.results.scores[1]
    stderr_metric = appended.metrics["stderr"]
    assert stderr_metric.value == pytest.approx(PLAIN_STDERR)
    assert stderr_metric.params == {}
