from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from aletheia.config import load_config
from aletheia.evolution import GenerationalEvolution


def _runtime_config() -> SimpleNamespace:
    cfg = load_config(Path('config.json'))
    cfg.runtime.device = 'cpu'
    cfg.runtime.evolution_population_size = 3
    cfg.runtime.evolution_candidate_mutations = ['baseline', 'model_noise', 'runtime_mutation']
    cfg.runtime.evolution_runtime_mutations = ['learning_rate']
    cfg.runtime.evolution_runtime_scales = [0.85]
    cfg.runtime.evolution_mutation_scale = 0.01
    cfg.runtime.evolution_min_improvement = 0.01
    cfg.runtime.evolution_max_metric_regression = 0.05
    cfg.runtime.evolution_benchmark_weights = {'language_bpb': 0.5, 'vision_ssl_loss': 0.5}
    cfg.runtime.evolution_keep_generations = 2
    cfg.runtime.evolution_auto_promote = True
    return cfg.runtime


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / 'project'
    (root / 'aletheia').mkdir(parents=True)
    (root / 'tests').mkdir(parents=True)
    (root / 'data' / 'browser_profile').mkdir(parents=True)
    (root / 'aletheia' / 'demo.py').write_text('value = 1\n', encoding='utf-8')
    (root / 'config.json').write_text(json.dumps({'runtime': {}}), encoding='utf-8')
    torch.save({'state_dict': {'weight': torch.ones(8, 8)}, 'optimizer': {'junk': 1}}, root / 'data' / 'aletheia.pt')
    torch.save({'vision': {'w': torch.ones(4)}, 'audio': {'w': torch.ones(4)}, 'object_discovery': {'w': torch.ones(4)}}, root / 'data' / 'perception.pt')
    (root / 'data' / 'speaker_memory.json').write_text('{"profiles": []}\n', encoding='utf-8')
    (root / 'data' / 'browser_profile' / 'Cookie').write_text('external-session-state', encoding='utf-8')
    return root


def test_full_state_clone_migrates_learned_state_but_not_browser_session(tmp_path):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    clone = tmp_path / 'clone'
    migrated = engine._copy_tree_with_learned_state(clone)
    assert (clone / 'data' / 'aletheia.pt').exists()
    assert (clone / 'data' / 'perception.pt').exists()
    assert (clone / 'data' / 'speaker_memory.json').exists()
    assert not (clone / 'data' / 'browser_profile').exists()
    assert 'data/aletheia.pt' in migrated
    assert 'data/perception.pt' in migrated


def test_checkpoint_mutation_preserves_schema_and_changes_weights(tmp_path):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    target = root / 'data' / 'aletheia.pt'
    before = torch.load(target, map_location='cpu', weights_only=True)['state_dict']['weight'].clone()
    result = engine._mutate_checkpoint(target, seed=11, scale=0.01)
    after = torch.load(target, map_location='cpu', weights_only=True)['state_dict']['weight']
    assert result['applied'] is True
    assert after.shape == before.shape
    assert not torch.equal(before, after)


def test_multimodal_score_normalizes_lower_is_better_metrics(tmp_path):
    engine = GenerationalEvolution(_make_project(tmp_path))
    runtime = SimpleNamespace(
        evolution_benchmark_weights={'language_bpb': 0.5, 'vision_ssl_loss': 0.5},
        evolution_max_metric_regression=0.05,
    )
    baseline = {'language_bpb': 2.0, 'vision_ssl_loss': 4.0}
    candidate = {'language_bpb': 1.8, 'vision_ssl_loss': 3.6}
    result = engine.score_candidate(baseline, candidate, runtime)
    assert result['score'] > 1.0
    assert result['hard_regression'] is False
    assert set(result['shared_metrics']) == {'language_bpb', 'vision_ssl_loss'}


def test_population_generation_promotes_best_candidate_and_records_lineage(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    runtime = _runtime_config()
    fake_agent = SimpleNamespace(config=SimpleNamespace(runtime=runtime), save=lambda: None)

    def benchmark(candidate_root, timeout=180, seed_override=None):
        manifest = candidate_root / 'data' / 'evolution_manifest.json'
        if not manifest.exists():
            return {'ok': True, 'metrics': {'language_bpb': 2.0, 'vision_ssl_loss': 4.0}}
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if payload['mutation'] == 'model_noise':
            return {'ok': True, 'metrics': {'language_bpb': 1.8, 'vision_ssl_loss': 3.8}}
        if payload['mutation'] == 'runtime_mutation':
            return {'ok': True, 'metrics': {'language_bpb': 2.0, 'vision_ssl_loss': 4.1}}
        return {'ok': True, 'metrics': {'language_bpb': 2.0, 'vision_ssl_loss': 4.0}}

    monkeypatch.setattr(engine, 'benchmark_candidate', benchmark)
    result = engine.run_generation(fake_agent, population_size=3, auto_promote=True)
    assert result['status'] == 'completed'
    assert result['promoted'] is True
    assert result['winner']['candidate_id'].endswith('model_noise')
    current = engine.current_generation()
    assert current['candidate_id'] == 'root'
    assert result['promotion']['status'] == 'pending_restart'
    assert result['restart_required'] is True
    assert (root / 'data' / 'evolution' / 'generations').exists()
    assert (root / 'data' / 'evolution' / 'backups').exists()


def test_no_promotion_when_candidate_has_hard_multimodal_regression(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    runtime = _runtime_config()
    fake_agent = SimpleNamespace(config=SimpleNamespace(runtime=runtime), save=lambda: None)

    def benchmark(candidate_root, timeout=180, seed_override=None):
        manifest = candidate_root / 'data' / 'evolution_manifest.json'
        if not manifest.exists():
            return {'ok': True, 'metrics': {'language_bpb': 2.0, 'vision_ssl_loss': 4.0}}
        return {'ok': True, 'metrics': {'language_bpb': 2.4, 'vision_ssl_loss': 5.0}}

    monkeypatch.setattr(engine, 'benchmark_candidate', benchmark)
    result = engine.run_generation(fake_agent, population_size=2, auto_promote=True)
    assert result['promoted'] is False
    assert result['winner'] is None
    assert engine.current_generation()['generation'] == 0


def test_lineage_metadata_is_json_and_generation_numbered(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    runtime = _runtime_config()
    fake_agent = SimpleNamespace(config=SimpleNamespace(runtime=runtime), save=lambda: None)
    monkeypatch.setattr(engine, 'benchmark_candidate', lambda *_a, seed_override=None, **_k: {'ok': True, 'metrics': {'language_bpb': 2.0, 'vision_ssl_loss': 4.0}})
    result = engine.run_generation(fake_agent, population_size=2, auto_promote=False)
    manifest_paths = list((root / 'data' / 'evolution' / 'generations' / f"generation_{result['generation']:05d}").glob('*/data/evolution_manifest.json'))
    assert len(manifest_paths) == 2
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['generation'] == result['generation']
        assert payload['parent'] == 'root'
        assert Path(payload['candidate_id']) .name == payload['candidate_id']


def test_promotion_is_staged_and_activated_at_restart(tmp_path):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    candidate = root / 'data' / 'evolution' / 'generations' / 'generation_00001' / 'g00001c00_model_noise'
    (candidate / 'data').mkdir(parents=True, exist_ok=True)
    (candidate / 'data' / 'demo_state.txt').write_text('candidate', encoding='utf-8')
    (root / 'data' / 'demo_state.txt').write_text('parent', encoding='utf-8')
    (candidate / 'config.json').write_text((root / 'config.json').read_text(encoding='utf-8'), encoding='utf-8')
    result = engine.promote_candidate(candidate, generation=1, candidate_id='g00001c00_model_noise', score={'score': 1.02})
    assert result['restart_required'] is True
    assert (root / 'data' / 'demo_state.txt').read_text(encoding='utf-8') == 'parent'
    applied = GenerationalEvolution.apply_pending_promotion(root)
    assert applied['applied'] is True
    assert (root / 'data' / 'demo_state.txt').read_text(encoding='utf-8') == 'candidate'
    assert applied['promotion']['status'] == 'active'


def test_promotion_rolls_back_on_filesystem_failure(tmp_path, monkeypatch):
    import os
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    candidate = root / 'data' / 'evolution' / 'generations' / 'generation_00001' / 'g00001c00_model_noise'
    (candidate / 'data').mkdir(parents=True, exist_ok=True)
    (candidate / 'data' / 'demo_state.txt').write_text('candidate', encoding='utf-8')
    (root / 'data' / 'demo_state.txt').write_text('parent', encoding='utf-8')
    (candidate / 'config.json').write_text((root / 'config.json').read_text(encoding='utf-8'), encoding='utf-8')
    engine.promote_candidate(candidate, generation=1, candidate_id='g00001c00_model_noise', score={'score': 1.02})
    original = os.replace
    calls = {'n': 0}
    def flaky(src, dst):
        calls['n'] += 1
        if calls['n'] == 2:
            raise OSError('simulated swap failure')
        return original(src, dst)
    monkeypatch.setattr('aletheia.evolution.os.replace', flaky)
    result = GenerationalEvolution.apply_pending_promotion(root)
    assert result['applied'] is False
    assert 'rolled_back' in result['status']
    assert (root / 'data' / 'demo_state.txt').read_text(encoding='utf-8') == 'parent'


def test_default_evolution_config_is_not_auto_promoting(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config.json')
    assert cfg.runtime.evolution_auto_promote is False
    assert set(cfg.runtime.evolution_benchmark_weights) == {
        'language_bpb', 'vision_ssl_loss', 'audio_ssl_loss', 'speech_cer', 'tts_holdout_loss'
    }


def test_incomplete_promotion_journal_is_recovered_before_retry(tmp_path):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    candidate = root / 'data' / 'evolution' / 'generations' / 'generation_00002' / 'g00002c00_model_noise'
    (candidate / 'data').mkdir(parents=True, exist_ok=True)
    (candidate / 'data' / 'demo_state.txt').write_text('candidate', encoding='utf-8')
    (root / 'data' / 'demo_state.txt').write_text('parent', encoding='utf-8')
    (candidate / 'config.json').write_text((root / 'config.json').read_text(encoding='utf-8'), encoding='utf-8')
    engine.promote_candidate(candidate, generation=2, candidate_id='g00002c00_model_noise', score={'score': 1.03})
    tx = root / 'data' / 'evolution' / 'promotion_transaction'
    tx.mkdir(parents=True, exist_ok=True)
    old = tx / 'old'; stage = tx / 'stage'
    old.mkdir(); stage.mkdir()
    # Simulate a process dying after moving the parent state out of the active tree.
    (old / 'demo_state.txt').write_text('parent', encoding='utf-8')
    (root / 'data' / 'demo_state.txt').unlink()
    journal = {
        'schema_version': 2,
        'status': 'applying',
        'active_data': str(root / 'data'),
        'old_dir': str(old),
        'stage_dir': str(stage),
        'staged_names': ['demo_state.txt'],
        'old_names': ['demo_state.txt'],
    }
    (tx / 'journal.json').write_text(json.dumps(journal), encoding='utf-8')
    result = GenerationalEvolution.apply_pending_promotion(root)
    assert result['applied'] is True
    assert (root / 'data' / 'demo_state.txt').read_text(encoding='utf-8') == 'candidate'


def test_browser_observation_signature_rejects_changed_element(tmp_path):
    from aletheia.browser import BrowserEnvironment, BrowserEnvironmentError
    class FakeWeb:
        class Config:
            allowed_schemes = ["http", "https"]
        config = Config()
        normalize_url = staticmethod(lambda url, base_url=None, allowed_schemes=None: url if str(url).startswith(("http://", "https://")) else None)
        _host_is_public = staticmethod(lambda host: True)
        _robots = staticmethod(lambda url: True)
    browser = BrowserEnvironment(tmp_path, FakeWeb(), headless=True, timeout_ms=5000)
    browser._ensure_started()
    try:
        browser.page.set_content("<button id='b'>Old</button>", wait_until='domcontentloaded')
        obs = browser.observe(screenshot=False)
        ref = obs.elements[0].ref
        sig = obs.elements[0].signature
        browser.page.set_content("<button id='b'>Changed</button>", wait_until='domcontentloaded')
        with pytest.raises(BrowserEnvironmentError):
            browser.click(ref, observation_id=obs.observation_id, signature=sig)
    finally:
        browser.close()


def test_generation_uses_paired_predeclared_seeds_for_each_candidate(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    runtime = _runtime_config()
    runtime.evolution_eval_seeds = [111, 222, 333]
    runtime.evolution_min_improvement = 0.005
    fake_agent = SimpleNamespace(config=SimpleNamespace(runtime=runtime), save=lambda: None)
    observed = []

    def benchmark(candidate_root, timeout=180, seed_override=None):
        observed.append((candidate_root.name, seed_override))
        manifest_path = candidate_root / "data" / "evolution_manifest.json"
        if not manifest_path.exists():
            metrics = {"language_bpb": 2.0, "vision_ssl_loss": 4.0}
        else:
            mutation = json.loads(manifest_path.read_text(encoding="utf-8"))["mutation"]
            if mutation == "model_noise":
                metrics = {"language_bpb": 1.8, "vision_ssl_loss": 3.8}
            else:
                metrics = {"language_bpb": 2.0, "vision_ssl_loss": 4.0}
        # Seed-varying values are deterministic and remain improved for the winner.
        delta = {111: 0.0, 222: 0.01, 333: -0.005}[seed_override]
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["mutation"] == "model_noise":
            metrics["language_bpb"] += delta
        return {"ok": True, "metrics": metrics}

    monkeypatch.setattr(engine, "benchmark_candidate", benchmark)
    result = engine.run_generation(fake_agent, population_size=3, auto_promote=False)
    assert result["status"] == "completed"
    assert result["eval_seeds"] == [111, 222, 333]
    assert result["baseline"]["seed_count"] == 3
    assert [item["seed"] for item in result["baseline"]["per_seed"]] == [111, 222, 333]
    assert all(len(item["benchmark"]["per_seed"]) == 3 for item in result["population"] if item["benchmark"]["ok"])
    assert set(seed for _name, seed in observed) == {111, 222, 333}
    assert all(item["score"].get("paired_seed_scores") for item in result["population"] if item["benchmark"]["ok"])


def test_generation_aborts_if_benchmark_adapter_cannot_apply_seed(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    engine = GenerationalEvolution(root)
    runtime = _runtime_config()
    fake_agent = SimpleNamespace(config=SimpleNamespace(runtime=runtime), save=lambda: None)
    monkeypatch.setattr(engine, "benchmark_candidate", lambda candidate_root, timeout=180: {"ok": True, "metrics": {"language_bpb": 1.0}})
    result = engine.run_generation(fake_agent, population_size=2, auto_promote=True)
    assert result["status"] == "aborted"
    assert result["reason"] == "benchmark_adapter_missing_seed_override"
    assert result["promoted"] is False
