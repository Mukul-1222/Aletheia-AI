from pathlib import Path

from aletheia.evolution import EvolutionLab


def test_hard_sandbox_is_fail_closed_when_linux_bubblewrap_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("aletheia.evolution.shutil.which", lambda _: None)
    assert EvolutionLab._hard_sandbox_prefix(tmp_path) is None


def test_hard_sandbox_does_not_claim_posix_bubblewrap_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("aletheia.evolution.os.name", "nt")
    monkeypatch.setattr("aletheia.evolution.sys.platform", "win32")
    monkeypatch.setattr("aletheia.evolution.shutil.which", lambda _: "bwrap.exe")
    assert EvolutionLab._hard_sandbox_prefix(tmp_path) is None
