"""Qwen backbone with a ten-action structured classification head.

``transformers``, ``peft`` and ``bitsandbytes`` are intentionally imported only
inside loading functions, so CPU-only metric/test environments can import this
module without those optional packages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..states.action_space import ACTIONS, NUM_ACTIONS
from .action_head import StructuredActionHead, last_non_padding_hidden


def _optional_hf_imports(*, need_peft: bool = False):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Loading a Qwen student requires the optional 'transformers' package"
        ) from exc
    peft_symbols = None
    if need_peft:
        try:
            from peft import (
                LoraConfig,
                PeftModel,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("QLoRA loading/training requires the optional 'peft' package") from exc
        peft_symbols = {
            "LoraConfig": LoraConfig,
            "PeftModel": PeftModel,
            "TaskType": TaskType,
            "get_peft_model": get_peft_model,
            "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        }
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, peft_symbols


def _resolve_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {value}") from exc


def _hidden_size(config: Any) -> int:
    candidates = [config, getattr(config, "text_config", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("hidden_size", "n_embd", "d_model"):
            value = getattr(candidate, attribute, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer the backbone hidden size from its config")


def _extract_last_hidden(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    hidden_stack = getattr(output, "hidden_states", None)
    if hidden_stack is None and isinstance(output, Mapping):
        hidden_stack = output.get("hidden_states")
    if hidden is None and hidden_stack is not None:
        hidden = hidden_stack[-1]
    if hidden is None:
        raise RuntimeError(
            "Backbone did not expose last_hidden_state; use a base transformer or "
            "enable output_hidden_states"
        )
    if hidden.ndim != 3:
        raise RuntimeError("Backbone hidden state must have shape [batch, sequence, hidden]")
    return hidden


class QwenActionPolicy(nn.Module):
    """Use Qwen's last non-padding hidden state to select one structured action."""

    action_names = ACTIONS

    def __init__(self, backbone: nn.Module, action_head: StructuredActionHead | None = None):
        super().__init__()
        self.backbone = backbone
        if action_head is None:
            config = getattr(backbone, "config", None)
            if config is None:
                raise ValueError("backbone.config is required when action_head is omitted")
            action_head = StructuredActionHead(_hidden_size(config))
        if action_head.classifier.out_features != NUM_ACTIONS:
            raise ValueError(f"The structured head must expose exactly {NUM_ACTIONS} actions")
        self.action_head = action_head

    def _transformer(self) -> nn.Module:
        base = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        # Qwen*ForCausalLM.model is the decoder without the vocabulary projection.
        return getattr(base, "model", base)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have shape [batch, sequence]")
        core = self._transformer()
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "use_cache": False,
        }
        output = core(**kwargs)
        hidden = _extract_last_hidden(output)
        pooled = last_non_padding_hidden(hidden, attention_mask)
        parameter = next(self.action_head.parameters())
        return pooled.to(device=parameter.device, dtype=parameter.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_action_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode(input_ids, attention_mask)
        output = self.action_head.policy(
            representation,
            valid_action_mask,
            temperature=temperature,
        )
        return {
            "logits": output.logits,
            "masked_logits": output.masked_logits,
            "probabilities": output.probabilities,
        }

    @torch.no_grad()
    def act(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_action_mask: torch.Tensor,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        probabilities = self(
            input_ids,
            attention_mask,
            valid_action_mask,
            temperature=temperature,
        )["probabilities"]
        if deterministic:
            return probabilities.argmax(dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)

    def save_policy(
        self,
        output_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.backbone, "save_pretrained"):
            raise TypeError("backbone must implement save_pretrained")
        self.backbone.save_pretrained(output / "qwen_adapter_or_backbone")
        torch.save(self.action_head.state_dict(), output / "action_head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(output / "tokenizer")
        payload = {
            **dict(metadata or {}),
            "actions": list(ACTIONS),
            "hidden_size": self.action_head.classifier.in_features,
        }
        (output / "policy_metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = True,
        torch_dtype: str | torch.dtype | None = "bfloat16",
        device_map: str | Mapping[str, Any] | None = "auto",
        trust_remote_code: bool = False,
        lora_config: Mapping[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> tuple["QwenActionPolicy", Any]:
        """Load Qwen, optionally prepare a trainable 4-bit LoRA adapter, and tokenizer."""

        need_peft = lora_config is not None
        AutoModel, AutoTokenizer, BitsAndBytesConfig, peft = _optional_hf_imports(need_peft=need_peft)
        dtype = _resolve_dtype(torch_dtype)
        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": bool(trust_remote_code),
            **model_kwargs,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            )
        backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        if lora_config is not None and peft is not None:
            if load_in_4bit:
                backbone = peft["prepare_model_for_kbit_training"](
                    backbone,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            config_values = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                **dict(lora_config),
            }
            config_values.setdefault("task_type", peft["TaskType"].CAUSAL_LM)
            backbone = peft["get_peft_model"](backbone, peft["LoraConfig"](**config_values))
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=bool(trust_remote_code),
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        policy = cls(backbone)
        try:
            input_device = backbone.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            input_device = next(backbone.parameters()).device
        if input_device.type != "meta":
            policy.action_head.to(input_device)
        return policy, tokenizer


def load_saved_policy(
    base_model_name: str,
    checkpoint_dir: str | Path,
    *,
    torch_dtype: str | torch.dtype | None = "bfloat16",
    device_map: str | Mapping[str, Any] | None = "auto",
    load_in_4bit: bool = True,
    trust_remote_code: bool = False,
) -> tuple[QwenActionPolicy, Any]:
    """Restore a saved PEFT adapter/backbone and its structured head."""

    checkpoint = Path(checkpoint_dir)
    metadata_path = checkpoint / "policy_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata.get("actions", ())) != ACTIONS:
        raise ValueError("Checkpoint action order does not match the fixed action vocabulary")
    policy, tokenizer = QwenActionPolicy.from_pretrained(
        base_model_name,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        device_map=device_map,
        lora_config=None,
        trust_remote_code=bool(trust_remote_code),
    )
    adapter_path = checkpoint / "qwen_adapter_or_backbone"
    # If this is a PEFT adapter, attach it to the freshly loaded base model.
    if (adapter_path / "adapter_config.json").is_file():
        _, _, _, peft = _optional_hf_imports(need_peft=True)
        # QLoRA training casts non-quantized normalization parameters to fp32.
        # Reproduce that dtype preparation before attaching the saved adapter;
        # gradient checkpointing is unnecessary for deployment.
        policy.backbone = peft["prepare_model_for_kbit_training"](
            policy.backbone,
            use_gradient_checkpointing=False,
        )
        policy.backbone = peft["PeftModel"].from_pretrained(policy.backbone, adapter_path)
    elif adapter_path.is_dir():
        AutoModel, _, _, _ = _optional_hf_imports(need_peft=False)
        dtype = _resolve_dtype(torch_dtype)
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": bool(trust_remote_code),
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        policy.backbone = AutoModel.from_pretrained(adapter_path, **kwargs)
    head_path = checkpoint / "action_head.pt"
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    policy.action_head.load_state_dict(state)
    try:
        head_device = policy.backbone.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        head_device = next(policy.backbone.parameters()).device
    if head_device.type != "meta":
        policy.action_head.to(head_device)
    tokenizer_dir = checkpoint / "tokenizer"
    if tokenizer_dir.is_dir():
        _, AutoTokenizer, _, _ = _optional_hf_imports(need_peft=False)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=bool(trust_remote_code)
        )
    return policy, tokenizer


LLMActionPolicy = QwenActionPolicy


__all__ = ["LLMActionPolicy", "QwenActionPolicy", "load_saved_policy"]
