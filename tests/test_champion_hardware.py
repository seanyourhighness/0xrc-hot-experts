import json
import struct

from exllamav3.champion.hardware import model_identity


def _write_model(root, payload):
    root.mkdir(exist_ok = True)
    (root / "config.json").write_text(json.dumps({
        "architectures": ["TestForCausalLM"],
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }))
    header = json.dumps({"weight": {"dtype": "I8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_model_fingerprint_hashes_weight_content(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = tmp_path / "model"
    _write_model(model, b"abcd")
    first = model_identity(model)
    assert model_identity(model)["fingerprint"] == first["fingerprint"]
    assert (tmp_path / "cache" / "exllamav3" / "champion-model-hashes-v1.json").is_file()
    _write_model(model, b"wxyz")
    second = model_identity(model)
    assert first["safetensors"][0]["header_sha256"] == second["safetensors"][0]["header_sha256"]
    assert first["safetensors"][0]["sha256"] != second["safetensors"][0]["sha256"]
    assert first["fingerprint"] != second["fingerprint"]
