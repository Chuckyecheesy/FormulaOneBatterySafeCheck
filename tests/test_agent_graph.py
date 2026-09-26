from formulatech.agents import build_race_graph, run_race_assessment


class FakeChatModel:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        class Response:
            content = "safe explanation"
        return Response()


def _graph():
    return build_race_graph(model_factory=lambda *args, **kwargs: FakeChatModel())


def test_overcharge_short_circuit():
    graph = _graph()
    state = run_race_assessment(
        {
            "voltage_v": 5.0,
            "current_a": -1.2,
            "temperature_c": 25.0,
            "duration_min": 60,
        },
        graph=graph,
    )

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert state["failures"][0]["code"] == "OVERCHARGED"
    assert [r["stage"] for r in state["stage_results"]] == [1]


def test_prediction_path_uses_template_when_ollama_unavailable(monkeypatch):
    graph = build_race_graph(model_factory=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ollama offline")))

    state = run_race_assessment(
        {
            "voltage_v": 3.9,
            "current_a": 0.5,
            "temperature_c": 25.0,
            "duration_min": 60,
        },
        graph=graph,
    )

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert any("template" in msg.lower() for msg in state["explanations"])
