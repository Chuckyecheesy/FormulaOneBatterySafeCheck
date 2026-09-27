"""End-to-end browser tests for the frontend with Playwright (spec/05-ui.md, issue #7).

Run with:  uv sync --group e2e
           uv run playwright install chromium
           uv run pytest tests/e2e            (add --headed to watch the browser)
The tests are skipped when Playwright is not installed. No trained model or Ollama is needed:
the model status, the prediction and the chat model are faked.
"""

import socket
import threading
import time

import pytest

pytest.importorskip("playwright")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import Page, expect  # noqa: E402

from formulatech import agents  # noqa: E402
from formulatech.agents import build_race_graph  # noqa: E402
from formulatech.api import create_app  # noqa: E402

CLEAN = {"voltage_v": "3.9", "current_a": "0.5", "temperature_c": "25", "duration_min": "60", "soc_percent": "50"}
MODEL_STATUS = {"version": "test", "gates_passed": True, "tolerance_accuracy": 95.0, "r2": 0.95,
                "r2_train": 0.99, "cv_mean": 0.95, "cv_std": 0.01}


class FakeChatModel:
    def invoke(self, prompt):
        class Response:
            content = "explanation"

        return Response()


@pytest.fixture(scope="module")
def prediction():
    """Validated model whose prediction each test can set (no trained model or Ollama needed)."""
    value = {"pct": 98.2}
    saved = agents.get_model_status, agents.predict_efficiency
    agents.get_model_status = lambda: MODEL_STATUS
    agents.predict_efficiency = lambda *a: value["pct"]
    yield value
    agents.get_model_status, agents.predict_efficiency = saved


@pytest.fixture(scope="module")
def base_url(prediction):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(graph=build_race_graph(model_factory=FakeChatModel))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page(base_url, page: Page, prediction):
    prediction["pct"] = 98.2
    page.goto(base_url + "/")
    expect(page.locator("h1")).to_have_text("Race Readiness Check")
    return page


def fill_and_submit(page: Page, **overrides):
    for name, value in {**CLEAN, **overrides}.items():
        page.fill(f"#{name}", value)
    expect(page.locator("#submit")).to_be_enabled()
    page.click("#submit")
    expect(page.locator("#result")).to_be_visible()


def body_background(page: Page) -> str:
    return page.evaluate("getComputedStyle(document.body).backgroundImage")


def assert_background(page: Page):
    assert "car-fire.svg" in body_background(page)
    assert page.evaluate("getComputedStyle(document.body).backgroundAttachment") == "fixed"


def assert_no_popup(page: Page):
    assert page.locator("#fail-popup, .popup, .popup-overlay, [role=dialog]").count() == 0


def test_background_svg_is_served(page: Page, base_url):
    response = page.request.get(base_url + "/car-fire.svg")
    assert response.ok
    assert "image/svg+xml" in response.headers["content-type"]
    assert "battery" in response.text()


def test_background_on_input_form(page: Page):
    assert_background(page)
    expect(page.locator("#result")).to_be_hidden()


def test_can_proceed_on_background(page: Page):
    fill_and_submit(page)
    expect(page.locator(".banner.ok h2")).to_have_text("✅ CAN PROCEED")
    expect(page.locator(".banner.ok")).to_contain_text("Predicted efficiency: 98.2 %")
    expect(page.locator("#stages li.passed")).to_have_count(3)
    assert_background(page)
    assert_no_popup(page)


def test_thermal_failure_has_no_popup(page: Page):
    """Issue #7: a thermal failure is shown on the page, with no popup."""
    fill_and_submit(page, temperature_c="30", duration_min="1")
    banner = page.locator("#result .banner.fail")
    expect(banner.locator("h2")).to_have_text("⛔ DO NOT PROCEED")
    expect(banner.locator("h3")).to_have_text("Stage 2: Thermal")
    expect(banner.locator(".failure")).to_have_count(3)
    expect(banner).to_contain_text("dT/dt = 0.08333 °C/s")
    expect(banner).to_contain_text("d²T/dt² = 0.001389 °C/s²")
    expect(banner).to_contain_text("T_chem = 5.00 °C")
    expect(banner.locator(".comment")).to_have_text(f"Comment: {agents.COMMENT_FIRE_RISK_NOW}")
    expect(page.locator("#stages li.failed")).to_have_text("Thermal")
    assert page.locator(".banner.fail").count() == 1  # rendered once, not duplicated into a dialog
    assert_background(page)
    assert_no_popup(page)
    # The form stays usable straight away: nothing covers it.
    page.fill("#temperature_c", "25")
    expect(page.locator("#submit")).to_be_enabled()


def test_overcharge_failure_on_page(page: Page):
    fill_and_submit(page, voltage_v="5", current_a="-1.2")
    banner = page.locator("#result .banner.fail")
    expect(banner.locator("h3")).to_have_text("Stage 1: Overcharge")
    expect(banner).to_contain_text("Battery already overcharged")
    expect(banner).not_to_contain_text("OVERCHARGED")
    assert_background(page)
    assert_no_popup(page)


def test_low_efficiency_failure_on_page(page: Page, prediction):
    prediction["pct"] = 65.0
    fill_and_submit(page)
    banner = page.locator("#result .banner.fail")
    expect(banner.locator("h3")).to_have_text("Stage 3: Prediction")
    expect(banner).to_contain_text("predicted efficiency = 65.00 %")
    expect(banner).to_contain_text("predicted efficiency ≥ 70 %")
    expect(banner.locator(".comment")).to_have_text(f"Comment: {agents.COMMENT_FIRE_RISK_IN_RACE}")
    assert_background(page)
    assert_no_popup(page)


def test_invalid_input_shows_field_error(page: Page):
    page.fill("#duration_min", "0")
    page.locator("#duration_min").blur()
    expect(page.locator("#duration_min-error")).to_have_text("Charging duration must be greater than 0 minutes.")
    expect(page.locator("#submit")).to_be_disabled()
    assert_background(page)


def test_screenshots(page: Page, prediction, tmp_path_factory):
    """Save one screenshot per view so the background can be checked by eye."""
    out = tmp_path_factory.getbasetemp() / "screenshots"
    out.mkdir(exist_ok=True)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.screenshot(path=out / "1-form.png", full_page=True)
    fill_and_submit(page)
    page.screenshot(path=out / "2-can-proceed.png", full_page=True)
    fill_and_submit(page, temperature_c="30", duration_min="1")
    page.screenshot(path=out / "3-thermal-fail.png", full_page=True)
