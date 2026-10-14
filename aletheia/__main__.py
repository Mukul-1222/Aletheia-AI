from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _requested_root() -> Path:
    """Resolve --root before loading the application so recovery follows the CLI target."""
    argv = sys.argv[1:]
    for index, value in enumerate(argv):
        if value == "--root" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
        if value.startswith("--root="):
            return Path(value.split("=", 1)[1]).resolve()
    return Path.cwd().resolve()


def _apply_pending_generation() -> None:
    """Activate staged learned state before runtime state is opened."""
    root = _requested_root()
    try:
        from .evolution import GenerationalEvolution
        result = GenerationalEvolution.apply_pending_promotion(root)
        if result.get("applied"):
            promotion = result.get("promotion", {})
            print(f"Aletheia activated promoted generation {promotion.get('generation')} / {promotion.get('candidate_id')}")
        elif result.get("status") not in {"clean", "recovered_previous_transaction"}:
            print(f"Aletheia generation activation: {result}")
    except Exception as exc:
        print(f"Aletheia generation activation failed: {exc!r}")


def _recover_failed_successor() -> None:
    """Keep a bad self-modified successor from making future starts unusable."""
    root = _requested_root()
    try:
        pending = root / "data" / "evolution" / "pending_successor.json"
        if not pending.exists():
            return
        # Import the application before accepting the successor as healthy.
        importlib.import_module("aletheia.cli")
        pending.unlink(missing_ok=True)
    except Exception:
        try:
            from .evolution import EvolutionLab
            result = EvolutionLab.recover_pending_startup(root)
            if result.get("recovered"):
                print("Aletheia recovered the previous working source after a failed successor startup.")
        except Exception as exc:
            print(f"Aletheia successor recovery failed: {exc!r}")


_apply_pending_generation()
_recover_failed_successor()

from .cli import main

if __name__ == "__main__":
    main()
