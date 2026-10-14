import ast
import json
from pathlib import Path


def test_source_has_no_pretrained_model_runtime_imports_or_loaders():
    root = Path(__file__).parents[1] / "aletheia"
    blocked_modules = {
        "transformers", "sentence_transformers", "ollama", "huggingface_hub",
        "vllm", "llama_cpp", "mlx_lm",
    }
    blocked_calls = {"from_pretrained", "AutoModel", "AutoTokenizer", "pipeline"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                top_level = name.split(".", 1)[0]
                assert top_level not in blocked_modules, f"external model runtime import in {path}: {name}"
            if isinstance(node, ast.Call):
                func = node.func
                call_name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                assert call_name not in blocked_calls, f"pretrained loader call in {path}: {call_name}"
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
    serialized = json.dumps(config).lower()
    assert "wikipedia" not in serialized
