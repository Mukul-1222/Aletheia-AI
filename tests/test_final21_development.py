import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from aletheia.browser import BrowserEnvironment, BrowserEnvironmentError
from aletheia.config import load_config, ModelConfig
from aletheia.model import AletheiaModel


HTML_PAGE = """
<!doctype html>
<html><head><title>FINAL21 Test Page</title></head>
<body>
  <main><h1>Interactive page</h1>
    <button id='button' onclick="document.getElementById('state').textContent='clicked'">Click me</button>
    <form onsubmit="event.preventDefault(); document.getElementById('state').textContent=document.getElementById('input').value">
      <input id='input' name='query' placeholder='type here'>
      <button type='submit'>Submit</button>
    </form>
    <div id='state'>initial</div>
  </main>
</body></html>
"""


def _fake_web():
    class FakeWeb:
        class Config:
            allowed_schemes = ["http", "https"]
        config = Config()
        normalize_url = staticmethod(lambda url, base_url=None, allowed_schemes=None: url if str(url).startswith(("http://", "https://")) else None)
        _host_is_public = staticmethod(lambda host: True)
        _robots = staticmethod(lambda url: True)
    return FakeWeb()


def _started_browser(tmp_path):
    browser = BrowserEnvironment(tmp_path, _fake_web(), headless=True, timeout_ms=5000, allow_form_submit=True)
    browser._ensure_started()
    browser.page.set_content(HTML_PAGE, wait_until="domcontentloaded")
    return browser


def test_browser_environment_dom_screenshot_and_actions(tmp_path):
    browser = _started_browser(tmp_path)
    try:
        obs = browser.observe()
        assert obs.title == "FINAL21 Test Page"
        assert obs.screenshot_b64
        refs = {item.text: item.ref for item in obs.elements}
        click_ref = next(ref for text, ref in refs.items() if "Click me" in text)
        clicked = browser.action({"action": "click", "ref": click_ref})
        assert "clicked" in clicked.text

        refreshed = browser.observe()
        input_ref = next(item.ref for item in refreshed.elements if item.name == "query")
        submit_ref = next(item.ref for item in refreshed.elements if item.text == "Submit")
        browser.action({"action": "type", "ref": input_ref, "text": "learn from the page"})
        submitted = browser.action({"action": "click", "ref": submit_ref})
        assert "learn from the page" in submitted.text
    finally:
        browser.close()


def test_browser_stale_reference_is_rejected(tmp_path):
    browser = _started_browser(tmp_path)
    try:
        obs = browser.observe()
        assert any(item.text == "Click me" for item in obs.elements)
        browser.observe()
        with pytest.raises(BrowserEnvironmentError):
            browser.click("e9999")
    finally:
        browser.close()


def test_final21_runtime_config_and_capability_present():
    cfg = load_config()
    assert cfg.runtime.browser_enabled is False
    assert cfg.runtime.browser_headless is True
    assert cfg.runtime.browser_dom_max_chars >= 1000
    assert cfg.runtime.browser_dom_max_elements >= 10
    assert "browser_interact" in cfg.capabilities


def test_final21_checkpoint_migrates_new_browser_capability(tmp_path):
    cfg = ModelConfig(dim=32, layers=1, heads=4, ffn=64, recurrent_dim=48, context=64, world_model_hidden=64, world_attention_heads=4)
    old_actions = ["discover", "search", "follow"]
    new_actions = old_actions + ["browser_interact"]
    old_model = AletheiaModel(cfg, old_actions)
    old_path = tmp_path / "old.pt"
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-3)
    old_model.save(old_path, old_optimizer)

    new_model = AletheiaModel(cfg, new_actions)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    restored = AletheiaModel.load(old_path, torch.device("cpu"), cfg, new_actions, new_optimizer)
    assert restored.action_names == new_actions
    for name in old_actions:
        old_idx = old_actions.index(name)
        new_idx = new_actions.index(name)
        assert torch.equal(
            old_model.action_embedding.weight[old_idx],
            restored.action_embedding.weight[new_idx],
        )


def test_final21_no_external_pretrained_model_imports():
    package = Path(__file__).parents[1] / "aletheia"
    banned = {"transformers", "sentence_transformers", "huggingface_hub", "ollama", "openai"}
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for module in banned:
            assert f"import {module}" not in text
            assert f"from {module} " not in text


def test_agent_browser_observation_enters_memory_and_workspace(tmp_path):
    from aletheia.agent import AletheiaAgent
    from aletheia.browser import BrowserObservation, BrowserElement

    cfg = load_config()
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.runtime.device = "cpu"
    cfg.runtime.browser_enabled = True
    cfg.runtime.memory_limit = 30
    cfg.runtime.minimum_bootstrap_memories = 1

    agent = AletheiaAgent(tmp_path, cfg)

    class FakeBrowser:
        def __init__(self):
            self.last_observation = BrowserObservation(
                url="https://example.invalid/",
                title="Observed page",
                text="Learned browser state",
                elements=[BrowserElement(ref="e1", tag="button", role="button", text="Action", aria_label="", name="", placeholder="", input_type="")],
                screenshot_b64="",
                timestamp=1.0,
                navigation_id=1,
            )
        def state_summary(self):
            return {"started": True, "url": self.last_observation.url, "title": self.last_observation.title, "elements": 1, "navigation_id": 1, "has_screenshot": False}
        def observe(self, screenshot=True):
            return self.last_observation
        def action(self, action):
            self.last_observation = BrowserObservation(
                url="https://example.invalid/",
                title="Observed page",
                text="After browser action",
                elements=self.last_observation.elements,
                screenshot_b64="",
                timestamp=2.0,
                navigation_id=2,
            )
            return self.last_observation
        def close(self):
            pass

    agent.browser.close()
    agent.browser = FakeBrowser()
    before = agent.state.detach().clone()
    result = agent.browser_observe()
    assert result["kind"] == "browser_observation"
    assert result["memory_id"] >= 1
    assert not torch.equal(before, agent.state)
    assert agent.memory.recent(1, kinds=("browser_observation",))
    agent.close()


def test_agent_browser_model_action_uses_current_dom_refs(tmp_path, monkeypatch):
    from aletheia.agent import AletheiaAgent
    from aletheia.browser import BrowserObservation, BrowserElement

    cfg = load_config()
    cfg.model.dim = 32
    cfg.model.layers = 1
    cfg.model.heads = 4
    cfg.model.ffn = 64
    cfg.model.recurrent_dim = 48
    cfg.model.context = 64
    cfg.model.world_model_hidden = 64
    cfg.model.world_attention_heads = 4
    cfg.runtime.device = "cpu"
    cfg.runtime.browser_enabled = True
    cfg.runtime.minimum_bootstrap_memories = 1
    agent = AletheiaAgent(tmp_path, cfg)
    try:
        obs = BrowserObservation(
            url="https://example.invalid/",
            title="Page",
            text="Button is visible",
            elements=[BrowserElement(ref="e7", tag="button", role="button", text="Do it", aria_label="", name="", placeholder="", input_type="")],
            timestamp=1.0,
            navigation_id=1,
        )
        monkeypatch.setattr(agent.model, "generate", lambda *args, **kwargs: '{"action":"click","ref":"e7"}')
        action = agent._browser_action_from_model(obs)
        assert action == {"action": "click", "ref": "e7"}
    finally:
        agent.close()
