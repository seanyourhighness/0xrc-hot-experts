from __future__ import annotations
from typing_extensions import override
import torch
import os
from ..util.device_copy import to_device
import weakref
from safetensors.torch import load_file

from ..model.config import Config
from ..model.model import Model
from ..modules import Embedding, Linear, GatedResidual
from ..modules.quant.exl3 import LinearEXL3
from ..modules.module import Module
from ..modules.arch_specific.qwen4_exp_mtp import Qwen4ExpMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen4_exp import Qwen4ExpConfig

"""
MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next: input combine over the trunk's
PRE-collapse hyper-connection stream stack (exported by the trunk's final mixer) plus the next
token's embedding, one full qwen4_exp decoder block (QSA attention, MoE, gated-residual sites),
and its own combine-less mixer. Shares the trunk's embedding and lm_head.

No reference implementation exists for this head; the input-combine stream handling
(Qwen4ExpMTPInputLayer.stream_tap) is a semantic guess that must be confirmed by acceptance
rate on the full model.
"""


class Qwen4ExpMTPStackOut(Module):
    """
    Terminal module of the MTP draft chain: passes the decoder block's stream stack through
    FLATTENED (bsz, seq, hc_mult * hidden) instead of collapsing it, so the model's forward
    output can feed the next drafting step's target_hidden (symmetric with the trunk's pre-mixer
    stack export). The mixer is owned here as a submodule (so it loads with the model) and is
    applied by sample_from_state() before the shared lm_head.
    """

    def __init__(self, config, key: str, mixer: GatedResidual):
        super().__init__(config, key, None)
        self.mixer = mixer
        self.register_submodule(mixer)

    def optimizer_targets(self):
        return []

    # The compile step collects a top-level module's output tensors by ITS key prefix; this
    # module's own key ("mtp_stack_out") names no tensors, so it must hand the collection to
    # the owned mixer (prefix "mtp.hyper_connection_mixer."), or the mixer's tensors stay in
    # the qtensors files and never reach the compiled shards
    def get_compile_sizes(self, stc):
        return self.mixer.get_compile_sizes(stc)

    def get_compile_tensors(self, stc):
        return self.mixer.get_compile_tensors(stc)

    def forward(self, x, params, out_dtype = None):
        return x.flatten(-2).half()


class Qwen4ExpMTPModel(Model):

    def __init__(
        self,
        config: Qwen4ExpConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .qwen4_exp import build_qwen4_block

        self.input_layer = Qwen4ExpMTPInputLayer(
            config = config,
            key = "mtp",
            hidden_size = config.hidden_size,
            hc_mult = config.hc_mult,
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        for idx in range(config.mtp_num_hidden_layers):
            self.modules.append(
                build_qwen4_block(
                    config,
                    f"mtp.layers.{idx}",
                    idx,
                    "full_attention",
                    qbits_key = "mtp_bits",
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # The draft chain's output is the flattened PRE-mixer stream stack (it feeds the next
        # drafting step's target_hidden); sample_from_state applies the mixer + shared lm_head
        self.stack_out = Qwen4ExpMTPStackOut(
            config,
            "mtp_stack_out",
            GatedResidual(
                config = config,
                key = "mtp.hyper_connection_mixer",
                hc_mult = config.hc_mult,
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                use_combine = False,
                out_dtype = torch.half,
            ),
        )
        self.modules.append(self.stack_out)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None
        self.mtp_sub_lm_head = None
        self.mtp_hot_vocab = 0
        self.mtp_hot_id_map = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and have the trunk's final mixer
        export the pre-collapse stream stack as the draft input state.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        hot_path = os.environ.get("EXL3_MTP_HOT_HEAD", "").strip()
        hot_groups_path = os.environ.get("EXL3_MTP_HOT_GROUPS", "").strip()
        if hot_path and hot_groups_path:
            raise ValueError("choose one of EXL3_MTP_HOT_HEAD or EXL3_MTP_HOT_GROUPS")
        if hot_groups_path:
            if target.loaded_tp:
                raise ValueError("MTP grouped head currently supports single-GPU layer split only")
            full_head = target.modules[-1].inner
            if not isinstance(full_head, LinearEXL3):
                raise ValueError("MTP grouped head requires an EXL3 target lm_head")
            with open(hot_groups_path, "r", encoding = "utf-8") as f:
                block_ids = [
                    int(line) for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            full_vocab = target.modules[-1].out_features_unpadded
            if not block_ids or block_ids != sorted(set(block_ids)):
                raise ValueError("MTP grouped head block IDs must be nonempty, unique, and sorted")
            if len(block_ids) % 8 or block_ids[-1] >= (full_vocab + 15) // 16:
                raise ValueError("MTP grouped head requires complete 128-token groups")
            for offset in range(0, len(block_ids), 8):
                first = block_ids[offset]
                if first % 8 or block_ids[offset:offset + 8] != list(range(first, first + 8)):
                    raise ValueError("MTP grouped head block IDs must be aligned groups of eight")
            block_idx = torch.tensor(block_ids, device = full_head.trellis.device, dtype = torch.long)
            token_ids = (
                block_idx[:, None] * 16 +
                torch.arange(16, device = block_idx.device)[None, :]
            ).flatten()
            if token_ids.numel() != len(block_ids) * 16 or token_ids.max() >= full_vocab:
                raise ValueError("MTP grouped head may not include a partial final vocabulary block")
            required = {x for x in target.config.eos_token_id_list if x is not None}
            if not required.issubset(set(token_ids.cpu().tolist())):
                raise ValueError("MTP grouped head omits required EOS token IDs")
            # This is an exact control: retain the original EXL3 quantization and Hadamard
            # metadata, slicing only the independently transformed 128-token groups.
            trellis = full_head.trellis.index_select(1, block_idx).contiguous()
            svh = full_head.svh.index_select(0, token_ids).contiguous()
            bias = full_head.bias.index_select(0, token_ids).contiguous() \
                if full_head.bias is not None else None
            self.mtp_sub_lm_head = LinearEXL3(
                config = target.config, in_features = full_head.in_features,
                out_features = token_ids.numel(), suh = full_head.suh, svh = svh,
                trellis = trellis, mcg = full_head.mcg_tensor, mul1 = full_head.mul1_tensor,
                bias = bias, out_dtype = full_head.out_dtype, key = "mtp.hot_grouped_lm_head",
            )
            embed = target_embed.embedding.weight.index_select(0, token_ids.cpu())
            self.input_layer.hot_embedding = embed.to(
                device = full_head.trellis.device, dtype = torch.float16
            ).contiguous()
            inverse = torch.full((full_vocab,), -1, device = token_ids.device, dtype = torch.long)
            inverse[token_ids] = torch.arange(token_ids.numel(), device = token_ids.device)
            self.input_layer.hot_inverse = inverse
            self.mtp_hot_id_map = token_ids
            self.mtp_hot_vocab = token_ids.numel()
        elif hot_path:
            if target.loaded_tp:
                raise ValueError("MTP selected head currently supports single-GPU layer split only")
            full_head = target.modules[-1].inner
            if not isinstance(full_head, LinearEXL3):
                raise ValueError("MTP selected head requires an EXL3 target lm_head")
            tensors = load_file(hot_path, device = str(full_head.trellis.device))
            token_ids = tensors.pop("token_ids").long().contiguous()
            embedding = tensors.pop("embedding").half().contiguous()
            if token_ids.numel() != embedding.shape[0] or token_ids.numel() % 128:
                raise ValueError("invalid MTP selected-head token/embedding geometry")
            if token_ids.unique().numel() != token_ids.numel() or token_ids.min() < 0 \
                    or token_ids.max() >= target.config.vocab_size:
                raise ValueError("invalid MTP selected-head token IDs")
            if embedding.shape[1] != full_head.in_features:
                raise ValueError("MTP selected-head embedding width does not match target head")
            required = {x for x in target.config.eos_token_id_list if x is not None}
            if not required.issubset(set(token_ids.cpu().tolist())):
                raise ValueError("MTP selected head omits required EOS token IDs")
            self.mtp_sub_lm_head = LinearEXL3(
                config = target.config, in_features = full_head.in_features, out_features = token_ids.numel(),
                suh = tensors.pop("suh"), svh = tensors.pop("svh"), trellis = tensors.pop("trellis"),
                mcg = tensors.pop("mcg", None), mul1 = tensors.pop("mul1", None),
                out_dtype = full_head.out_dtype, key = "mtp.hot_lm_head",
            )
            if tensors:
                raise ValueError(f"unexpected MTP selected-head tensors: {sorted(tensors)}")
            inverse = torch.full((target.config.vocab_size,), -1, device = token_ids.device, dtype = torch.long)
            inverse[token_ids] = torch.arange(token_ids.numel(), device = token_ids.device)
            self.input_layer.hot_embedding = embedding
            self.input_layer.hot_inverse = inverse
            self.mtp_hot_id_map = token_ids
            self.mtp_hot_vocab = token_ids.numel()

        target_mixer = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_mixer, GatedResidual) and not target_mixer.use_combine, \
            "Expected the trunk's combine-less mixer immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_mixer.key},
        })

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        # state is the flattened pre-mixer stream stack; collapse it before the shared head
        mixer = self.stack_out.mixer
        bsz, seq, _ = state.shape
        stack = to_device(state, mixer.device).view(bsz, seq, mixer.hc_mult, mixer.hidden_size)
        state = mixer.forward(stack, params)
        if self.mtp_sub_lm_head is not None:
            logits = self.mtp_sub_lm_head.forward(state, params)
            if params.get("export_draft_conf"):
                conf, sub_ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
            else:
                sub_ids = torch.argmax(logits, dim = -1)
            return self.mtp_hot_id_map[sub_ids]
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
