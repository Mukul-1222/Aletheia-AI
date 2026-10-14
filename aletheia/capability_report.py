"""Produce an honest optional-capability inventory for the current host.

Dependency presence is not treated as an end-to-end capability test. The default
report therefore uses ``not tested`` unless required packages are missing. Run actual
hardware/data tests separately before changing a capability to ``tested``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

STATUS = {"tested", "not tested", "dependency missing", "failed"}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def build_capability_report(root: Path | None = None, *, release: str = "FINAL34") -> dict:
    root = (Path(root) if root is not None else Path.cwd()).resolve()
    modules = {
        "torch": _has_module("torch"),
        "PIL": _has_module("PIL"),
        "cv2": _has_module("cv2"),
        "sounddevice": _has_module("sounddevice"),
        "playwright": _has_module("playwright"),
        "pypdf": _has_module("pypdf"),
        "requests": _has_module("requests"),
        "bs4": _has_module("bs4"),
    }
    chromium = next((shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "ms-playwright") if shutil.which(name)), None)
    voice_command = "powershell" if platform.system() == "Windows" and shutil.which("powershell") else (
        "say" if platform.system() == "Darwin" and shutil.which("say") else ("espeak" if shutil.which("espeak") else None)
    )
    config_payload: dict = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                config_payload = loaded_config.get("runtime", loaded_config)
                if not isinstance(config_payload, dict):
                    config_payload = {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            config_payload = {}

    def entry(name: str, requirements: list[str], details: str) -> dict:
        missing = [item for item in requirements if not modules.get(item, False)]
        status = "dependency missing" if missing else "not tested"
        return {"name": name, "status": status, "required_dependencies": requirements,
                "missing_dependencies": missing, "details": details}

    capabilities = [
        entry("local_language_model", ["torch"], "Model load/generation and quality are not tested by this inventory."),
        entry("vision_image_processing", ["torch", "PIL", "cv2"], "Image and live-camera paths require separate smoke and held-out tests."),
        entry("audio_capture_and_playback", ["torch", "sounddevice"], "Microphone permission, device I/O, sample rates, and streaming are not tested."),
        entry("speech_recognition", ["torch"], "Held-out CER/WER is not tested; dependency presence is insufficient."),
        entry("learned_tts", ["torch"], "Audio/text dataset, checkpoint readiness, intelligibility, and latency are not tested."),
        entry("documents", ["pypdf", "PIL"], "Representative PDF/image/DOCX/EPUB inputs are not tested by this inventory."),
        entry("web_fetch", ["requests", "bs4"], "Network policy, provenance, parsing quality, and hostile content are not tested."),
        entry("browser", ["playwright"], "Chromium startup, HTTP guards, WebSocket blocking, and navigation security require end-to-end tests."),
    ]
    if not voice_command:
        capabilities.append({"name": "host_voice_output", "status": "dependency missing", "required_dependencies": [],
                             "missing_dependencies": ["native speech command"], "details": "No supported host TTS command was detected."})
    else:
        capabilities.append({"name": "host_voice_output", "status": "not tested", "required_dependencies": [],
                             "missing_dependencies": [], "details": f"Detected command {voice_command!r}; actual playback is not tested."})
    browser_capability = next(item for item in capabilities if item["name"] == "browser")
    browser_enabled = bool(config_payload.get("browser_enabled", False))
    allow_domains = config_payload.get("browser_allow_domains", [])
    proxy_url = str(config_payload.get("browser_egress_proxy_url", "")).strip()
    proxy_attested = bool(config_payload.get("browser_egress_proxy_policy_confirmed", False))
    browser_capability["details"] += f" Detected browser executable: {chromium or 'none'}; this does not confirm Playwright can launch it."
    browser_capability["browser_enabled_in_config"] = browser_enabled
    browser_capability["domain_allowlist_count"] = len(allow_domains) if isinstance(allow_domains, list) else 0
    browser_capability["egress_proxy_configured"] = bool(proxy_url)
    browser_capability["egress_policy_attested"] = proxy_attested
    if not browser_enabled:
        browser_capability["details"] += " Browser capability is disabled by configuration."
    elif not isinstance(allow_domains, list) or not allow_domains:
        browser_capability["status"] = "failed"
        browser_capability["details"] += " Browser is enabled but has no explicit domain allowlist; external navigation must fail closed."
    elif not proxy_url or not proxy_attested:
        browser_capability["status"] = "failed"
        browser_capability["details"] += " Browser is enabled without an attested policy-enforcing egress proxy; external navigation must fail closed."
    else:
        browser_capability["details"] += " Proxy policy is operator-attested, not independently verified by this inventory."
    for item in capabilities:
        if item["status"] not in STATUS:
            raise AssertionError(f"invalid capability status: {item['status']}")
    return {
        "schema_version": 1,
        "release": str(release),
        "generated_at_unix": time.time(),
        "root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "modules": modules,
        "capabilities": capabilities,
        "policy": "A dependency check is not an end-to-end test; no capability is marked tested by this report alone.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Aletheia project root")
    parser.add_argument("--output", default="data/capability_report.json", help="JSON output path, relative to --root by default")
    parser.add_argument("--release", default="FINAL34", help="release label to record in the report")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    report = build_capability_report(root, release=args.release)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "capabilities": report["capabilities"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
