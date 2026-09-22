"""Regression coverage for profile-seeded dynamic CPU-MoE placement."""
from types import SimpleNamespace

import torch

from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU


class _Stc:
    def __init__(self, tensors):
        self.tensors = tensors

    def list_tensors(self, key):
        return {key + ".trellis": {"shape": [1, 1, 16]}}

    def get_tensor(self, key, _device, optional = False, float2half = False):
        if optional and key not in self.tensors:
            return None
        return self.tensors[key].clone()


class _Host:
    def __init__(self):
        self.aux = {}
        self.installs = []

    def install_expert(self, layer_idx, cpu_local, keys):
        self.installs.append((layer_idx, cpu_local, list(keys)))


def _expert(key):
    return SimpleNamespace(key = key)


def _gpu_slot():
    return SimpleNamespace(inner = SimpleNamespace(
        trellis = torch.full((1, 1, 16), -1, dtype = torch.int16),
        suh = torch.full((1,), -1.0, dtype = torch.float16),
        svh = torch.full((1,), -1.0, dtype = torch.float16),
        bias = None,
    ))


def test_seeded_swap_translates_router_ids_to_checkpoint_ids():
    # Router IDs [0, 1, 2, 3] refer to checkpoint experts [2, 0, 3, 1].
    full_u = [_expert(f"u{index}") for index in range(4)]
    full_d = [_expert(f"d{index}") for index in range(4)]
    tensors = {}
    for index in range(4):
        for prefix, base in (("u", 100), ("d", 200)):
            tensors[f"{prefix}{index}.trellis"] = torch.full(
                (1, 1, 16), base + index, dtype = torch.int16
            )
            tensors[f"{prefix}{index}.suh"] = torch.full((1,), base + index, dtype = torch.float16)
            tensors[f"{prefix}{index}.svh"] = torch.full((1,), base + index, dtype = torch.float16)

    host = _Host()
    module = BlockSparseMLP_CPU.__new__(BlockSparseMLP_CPU)
    module.config = SimpleNamespace(stc = _Stc(tensors))
    module._split_perm = [2, 0, 3, 1]
    module._split_saved = (None, full_u, full_d)
    module.gated = False
    module.ups = [_gpu_slot(), _gpu_slot()]
    module.downs = [_gpu_slot(), _gpu_slot()]
    module.cpu_split_first = 2
    module.cpu_host = host
    module.cpu_layer_idx = 17

    placement = torch.tensor([0, 1, 2, 3], dtype = torch.long)
    assert module._split_swap_experts(0, 2, placement)
    assert module.ups[0].inner.trellis.flatten()[0].item() == 103
    assert module.downs[0].inner.trellis.flatten()[0].item() == 203
    assert host.installs == [(17, 0, ["u2", "d2"])]
    assert placement.tolist() == [2, 1, 0, 3]
