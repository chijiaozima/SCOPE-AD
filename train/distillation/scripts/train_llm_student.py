#!/usr/bin/env python3
"""Train Qwen QLoRA policies with a masked ten-action structured head."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distil planner policies into Qwen with 4-bit LoRA and a ten-action head. "
            "Only leakage-checked train prompts fit weights; validation selects checkpoints."
        )
    )
    parser.add_argument("--config", required=True, help="Distillation YAML configuration.")
    return parser


def _records(frame: Any, *, prompt_max_chars: int) -> list[dict[str, Any]]:
    import numpy as np
    from src.students.prompts import build_structured_prompt, validate_prompt_schema

    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            prompt_state = json.loads(str(row["prompt_state_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prompt_state_json at row {index}") from exc
        safe_state = validate_prompt_schema(prompt_state)
        rebuilt_prompt = build_structured_prompt(
            prompt_state, max_chars=prompt_max_chars
        )
        stored_prompt = str(row["prompt"])
        if stored_prompt != rebuilt_prompt:
            raise ValueError(
                f"Stored prompt does not exactly match prompt_state_json at row {index}"
            )
        try:
            recorded_character_count = int(row["prompt_char_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prompt_char_count at row {index}") from exc
        if recorded_character_count != len(rebuilt_prompt):
            raise ValueError(f"prompt_char_count mismatch at row {index}")
        full_prompt = build_structured_prompt(prompt_state)
        recorded_truncated = row["prompt_observations_truncated"]
        if not isinstance(recorded_truncated, (bool, np.bool_)):
            raise ValueError(
                f"prompt_observations_truncated must be boolean at row {index}"
            )
        if bool(recorded_truncated) != (rebuilt_prompt != full_prompt):
            raise ValueError(
                f"prompt_observations_truncated mismatch at row {index}"
            )
        mask = np.asarray(row["valid_action_mask"], dtype=bool)
        target = np.asarray(row["planner_policy"], dtype=np.float32)
        action = int(row["chosen_action_index"])
        if mask.shape != (10,) or target.shape != (10,):
            raise ValueError(f"Student record {index} does not contain ten actions")
        if (
            not np.isfinite(target).all()
            or np.any(target < 0)
            or np.any(target[~mask] > 1e-7)
            or not np.isclose(target.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError(f"Student record {index} has an invalid teacher policy")
        if not np.array_equal(mask, np.asarray(safe_state.valid_action_mask, dtype=bool)):
            raise ValueError(
                f"Student mask does not match the validated prompt state at row {index}"
            )
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            raise ValueError(f"Student record {index} selects an invalid target action")
        if action != int(target.argmax()):
            raise ValueError(
                f"Student target action does not match teacher-policy argmax at row {index}"
            )
        records.append(
            {
                # Never pass the stored string to the tokenizer. It is checked above
                # only as an integrity assertion; this canonical rebuild is the input.
                "prompt": rebuilt_prompt,
                "prompt_state": prompt_state,
                "mask": mask,
                "target": target,
                "action": action,
            }
        )
    return records


def _lora_target_modules(value: Any) -> str | list[str]:
    """Preserve PEFT's special all-linear token or validate an explicit list."""

    if value is None:
        return list(DEFAULT_LORA_TARGET_MODULES)
    if isinstance(value, str):
        target = value.strip()
        if not target:
            raise ValueError("llm.target_modules cannot be empty")
        return target
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("llm.target_modules must be 'all-linear' or a sequence of names")
    modules = [str(module).strip() for module in value]
    if not modules or any(not module for module in modules):
        raise ValueError("llm.target_modules must contain non-empty module names")
    if len(set(modules)) != len(modules):
        raise ValueError("llm.target_modules cannot contain duplicates")
    return modules


def _collator(tokenizer: Any, max_length: int, torch: Any):
    import numpy as np

    def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [str(record["prompt"]) for record in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "valid_action_mask": torch.from_numpy(
                np.stack([record["mask"] for record in batch]).astype(np.bool_)
            ),
            "target_policy": torch.from_numpy(
                np.stack([record["target"] for record in batch]).astype(np.float32)
            ),
            "target_action": torch.as_tensor(
                [record["action"] for record in batch], dtype=torch.long
            ),
        }

    return collate


def _model_input_device(policy: Any, torch: Any) -> Any:
    for parameter in policy.backbone.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_token_lengths(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> int:
    """Prove the defensive collator truncation will never remove prompt fields."""

    longest = 0
    for start in range(0, len(records), 128):
        prompts = [str(record["prompt"]) for record in records[start : start + 128]]
        encoded = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths = [len(values) for values in encoded["input_ids"]]
        longest = max(longest, max(lengths, default=0))
    if longest > int(max_length):
        raise ValueError(
            f"A leakage-safe prompt requires {longest} tokens, exceeding llm.max_length="
            f"{max_length}; refusing to silently truncate mandatory fields"
        )
    return longest


def _resolve_base_model_snapshot(
    model_id: str,
    *,
    requested_revision: str | None,
    local_files_only: bool,
) -> tuple[Path, str, str]:
    """Resolve the base model once, then load every seed from one immutable snapshot."""

    local = Path(model_id).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"llm.model is not a directory: {local}")
        revision = str(requested_revision or "")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(
                "A local llm.model requires llm.revision as an immutable 40-hex commit"
            )
        return local.resolve(), revision, "configured_local_snapshot"

    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Resolving a pinned Qwen snapshot requires the huggingface_hub package"
        ) from exc
    if re.fullmatch(r"[0-9a-f]{40}", requested_revision or "") is not None:
        resolved_revision = str(requested_revision)
        resolution = "configured_immutable_hub_commit"
    else:
        if local_files_only:
            raise ValueError(
                "llm.local_files_only=true requires an immutable 40-hex llm.revision"
            )
        info = model_info(model_id, revision=requested_revision or None)
        resolved_revision = str(getattr(info, "sha", ""))
        resolution = "hub_revision_resolved_before_load"
    if re.fullmatch(r"[0-9a-f]{40}", resolved_revision) is None:
        raise ValueError(
            f"Hugging Face did not resolve {model_id!r} to an immutable commit SHA"
        )
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=resolved_revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    return snapshot, resolved_revision, resolution


def _loss(
    policy: Any,
    batch: Mapping[str, Any],
    *,
    input_device: Any,
    head_device: Any,
    objective: str,
    temperature: float,
    hard_weight: float,
    return_probabilities: bool = False,
) -> tuple[Any, float, float] | tuple[Any, float, float, Any]:
    from src.students.losses import distillation_loss, hard_action_loss

    output = policy(
        batch["input_ids"].to(input_device),
        batch["attention_mask"].to(input_device),
        batch["valid_action_mask"].to(head_device),
        temperature=1.0,
    )
    logits = output["logits"]
    mask = batch["valid_action_mask"].to(logits.device)
    action = batch["target_action"].to(logits.device)
    if objective == "hard_action":
        hard = hard_action_loss(logits, action, mask)
        result = (hard, 0.0, float(hard.detach()))
        return (*result, output["probabilities"]) if return_probabilities else result
    components = distillation_loss(
        logits,
        batch["target_policy"].to(logits.device),
        mask,
        target_action=action,
        hard_weight=hard_weight,
        temperature=temperature,
    )
    result = (
        components.loss,
        float(components.soft_kl.detach()),
        float(components.hard_nll.detach()),
    )
    return (*result, output["probabilities"]) if return_probabilities else result


def _restore_trainable_parameters(
    policy: Any,
    best_trainable: Mapping[str, Any],
    torch: Any,
) -> None:
    """Restore LoRA/head tensors without materialising the full backbone state."""

    current = {
        name: parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(best_trainable):
        missing = sorted(set(current) - set(best_trainable))
        extra = sorted(set(best_trainable) - set(current))
        raise ValueError(
            f"Best trainable checkpoint keys differ; missing={missing}, extra={extra}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(
                best_trainable[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def run(config: Mapping[str, Any]) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import pandas as pd
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader

    from scripts.cache_subset_states import (
        ArtifactGateError,
        artifact_value,
        canonical_mapping_sha256,
        directory_payload_sha256,
        load_config_file,
        load_sidecar,
        metadata_path,
        planner_contract,
        require_columns,
        require_file,
        resolve_artifact,
        sha256,
        utc_now,
        write_audit,
        write_json_atomic,
    )
    from src.states.action_space import ACTIONS
    from src.students.llm_action_policy import QwenActionPolicy
    from src.students.prompts import assert_no_forbidden_fields
    from src.utils.seed import set_seed

    trajectory_path = resolve_artifact(config, artifact_value(config, "planner_trajectories"))
    checkpoint_root = resolve_artifact(config, artifact_value(config, "llm_checkpoint_dir"))
    require_file(trajectory_path, description="planner distillation trajectories")
    trajectory_metadata = load_sidecar(
        trajectory_path,
        expected_kind="planner_distillation_trajectories",
        require_deployable=False,
        require_privileged=False,
    )
    required_gates = {
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "test_used_for_training_or_selection": False,
        "offline_training_only": True,
        "student_input_projection_deployable": True,
        "budget_conditioned": True,
    }
    for key, expected in required_gates.items():
        if trajectory_metadata.get(key) != expected:
            raise ArtifactGateError(
                f"Trajectory metadata gate {key} expected {expected!r}"
            )
    for digest_field in (
        "h_schema_sha256",
        "h_implementation_sha256",
        "h_input_schema_sha256",
        "h_input_vectors_sha256",
    ):
        if not trajectory_metadata.get(digest_field):
            raise ArtifactGateError(f"Trajectory artifact lacks canonical {digest_field}")
    canonical_provenance = {
        field: trajectory_metadata[field]
        for field in (
            "h_schema_sha256",
            "h_implementation_sha256",
            "h_input_schema_sha256",
            "h_input_vectors_sha256",
        )
    }
    for digest_field in (
        "source_subset_sha256",
        "source_prepared_sha256",
        "source_preprocessor_sha256",
    ):
        if not trajectory_metadata.get(digest_field):
            raise ArtifactGateError(f"Trajectory artifact lacks {digest_field} provenance")
    training_budgets = trajectory_metadata.get("training_budgets_usd")
    if not isinstance(training_budgets, list) or not training_budgets:
        raise ArtifactGateError("Trajectory artifact lacks training_budgets_usd")
    planner_source_config = (
        config
        if isinstance(config.get("planner"), Mapping)
        else load_config_file(str(config.get("planner_config", "configs/planner.yaml")))
    )
    expected_planner_contract = planner_contract(planner_source_config)
    expected_planner_contract_sha256 = canonical_mapping_sha256(
        expected_planner_contract
    )
    if (
        trajectory_metadata.get("planner_contract") != expected_planner_contract
        or trajectory_metadata.get("planner_contract_sha256")
        != expected_planner_contract_sha256
    ):
        raise ArtifactGateError(
            "Teacher trajectories use a different resolved planner contract"
        )
    teacher_type = str(trajectory_metadata.get("teacher_type", "soft_ebm"))
    if teacher_type not in {"soft_ebm", "exact_soft_bellman"}:
        raise ArtifactGateError(f"Unsupported trajectory teacher_type={teacher_type!r}")
    teacher_manifest = require_file(
        trajectory_metadata.get("teacher_manifest", ""),
        description="trajectory teacher planner manifest",
    )
    teacher_manifest_sha256 = sha256(teacher_manifest)
    if trajectory_metadata.get("teacher_manifest_sha256") != teacher_manifest_sha256:
        raise ArtifactGateError("Teacher trajectory planner-manifest digest mismatch")
    teacher_provenance = {
        "teacher_type": teacher_type,
        "planner_contract": expected_planner_contract,
        "planner_contract_sha256": expected_planner_contract_sha256,
        "teacher_manifest": str(teacher_manifest),
        "teacher_manifest_sha256": teacher_manifest_sha256,
    }
    prompt_max_chars_value = trajectory_metadata.get("prompt_max_chars")
    if (
        isinstance(prompt_max_chars_value, bool)
        or not isinstance(prompt_max_chars_value, int)
        or prompt_max_chars_value <= 0
    ):
        raise ArtifactGateError(
            "Trajectory metadata lacks a positive integer prompt_max_chars"
        )
    prompt_max_chars = int(prompt_max_chars_value)

    distillation = config.get("distillation")
    llm = config.get("llm")
    if not isinstance(distillation, Mapping) or not isinstance(llm, Mapping):
        raise ArtifactGateError("distillation and llm mappings are required")
    max_length = int(llm.get("max_length", 768))
    if max_length <= 0:
        raise ArtifactGateError("llm.max_length must be positive")
    expected_prompt_max_chars = min(3000, max_length * 4)
    if prompt_max_chars != expected_prompt_max_chars:
        raise ArtifactGateError(
            "Trajectory prompt_max_chars does not match the current verified llm.max_length: "
            f"{prompt_max_chars} != {expected_prompt_max_chars}"
        )
    frame = pd.read_parquet(trajectory_path)
    require_columns(
        frame,
        (
            "teacher_seed",
            "patient_id",
            "split",
            "budget_usd",
            "prompt_state_json",
            "prompt",
            "prompt_char_count",
            "prompt_observations_truncated",
            "valid_action_mask",
            "planner_policy",
            "chosen_action_index",
        ),
        artifact=trajectory_path,
    )
    if set(frame["split"]) != {"train", "validation"}:
        raise ArtifactGateError("LLM training artifact must contain train/validation only")
    forbidden = {"true_label", "label", "oracle_q", "hidden_modalities"} & set(frame.columns)
    if forbidden:
        raise ArtifactGateError(f"Forbidden LLM data columns: {sorted(forbidden)}")
    for value in frame["prompt_state_json"]:
        assert_no_forbidden_fields(json.loads(str(value)))
    if llm.get("use_action_head") is not True:
        raise ArtifactGateError("LLM must use the structured ten-action head")
    objective = str(llm.get("train_objective", distillation.get("objective", "soft_kl")))
    if objective not in {"soft_kl", "hard_action", "soft_kl_with_hard"}:
        raise ArtifactGateError(f"Unsupported LLM objective: {objective}")
    temperature = float(distillation.get("temperature", 1.0))
    hard_weight = float(distillation.get("hard_action_weight", 0.0))
    if objective == "soft_kl_with_hard" and hard_weight <= 0:
        raise ArtifactGateError("soft_kl_with_hard requires hard_action_weight > 0")
    if temperature <= 0 or hard_weight < 0:
        raise ArtifactGateError("Distillation temperature/weight is invalid")
    model_id = str(llm.get("model", "Qwen/Qwen3.5-9B"))
    trust_remote_code = bool(llm.get("trust_remote_code", False))
    if llm.get("load_in_4bit") is not True:
        raise ArtifactGateError("Qwen3.5-9B student training requires llm.load_in_4bit=true")
    resolved_target_modules = _lora_target_modules(llm.get("target_modules"))
    torch_dtype = str(llm.get("torch_dtype", "bfloat16"))
    training_device_map = llm.get("device_map")
    if training_device_map != {"": 0}:
        raise ArtifactGateError(
            "Single-4090 QLoRA training requires llm.device_map={'': 0}; "
            "automatic CPU/disk offload is not allowed"
        )
    quantization_recipe = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch_dtype,
    }
    requested_revision_value = llm.get("revision")
    requested_revision = (
        None if requested_revision_value is None else str(requested_revision_value)
    )
    if trust_remote_code and re.fullmatch(r"[0-9a-f]{40}", requested_revision or "") is None:
        raise ArtifactGateError(
            "trust_remote_code=true requires llm.revision as an immutable 40-hex commit"
        )
    teacher_seeds = sorted(int(value) for value in frame["teacher_seed"].unique())
    paired_seed42 = (
        isinstance(config.get("ablation"), Mapping)
        and config["ablation"].get("paired_seed42_ablation") is True
        and teacher_seeds == [42]
    )
    if len(teacher_seeds) < 3 and not paired_seed42:
        raise ArtifactGateError("At least three teacher seeds are required")
    records_by_seed_split: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in teacher_seeds:
        seed_frame = frame.loc[frame["teacher_seed"] == seed]
        for split in ("train", "validation"):
            records_by_seed_split[(seed, split)] = _records(
                seed_frame.loc[seed_frame["split"] == split],
                prompt_max_chars=prompt_max_chars,
            )
        if not records_by_seed_split[(seed, "train")] or not records_by_seed_split[
            (seed, "validation")
        ]:
            raise ArtifactGateError(
                f"Teacher seed {seed} lacks train or validation prompts"
            )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_seed_dirs = [
        checkpoint_root / f"seed_{seed}"
        for seed in teacher_seeds
        if (checkpoint_root / f"seed_{seed}").exists()
    ]
    if existing_seed_dirs:
        raise ArtifactGateError(
            "Refusing to merge Qwen payloads into existing seed directories; choose a "
            f"fresh output directory: {[str(path) for path in existing_seed_dirs]}"
        )
    base_model_snapshot, resolved_revision, revision_resolution = (
        _resolve_base_model_snapshot(
            model_id,
            requested_revision=requested_revision,
            local_files_only=bool(llm.get("local_files_only", False)),
        )
    )
    checkpoint_records: list[dict[str, Any]] = []
    maximum_prompt_tokens = 0

    for seed in teacher_seeds:
        set_seed(seed, deterministic=True)
        train_records = records_by_seed_split[(seed, "train")]
        validation_records = records_by_seed_split[(seed, "validation")]
        lora_config = {
            "r": int(llm.get("lora_rank", 8)),
            "lora_alpha": int(llm.get("lora_alpha", 16)),
            "lora_dropout": float(llm.get("lora_dropout", 0.05)),
            "target_modules": resolved_target_modules,
        }
        policy, tokenizer = QwenActionPolicy.from_pretrained(
            str(base_model_snapshot),
            load_in_4bit=bool(llm.get("load_in_4bit", True)),
            torch_dtype=torch_dtype,
            device_map=training_device_map,
            trust_remote_code=trust_remote_code,
            lora_config=lora_config,
        )
        input_device = _model_input_device(policy, torch)
        if input_device.type != "cuda":
            raise ArtifactGateError(
                f"Qwen training backbone was not placed on CUDA: {input_device}"
            )
        policy.action_head.to(input_device)
        head_device = next(policy.action_head.parameters()).device
        observed_prompt_tokens = _validate_token_lengths(
            tokenizer,
            [*train_records, *validation_records],
            max_length=max_length,
        )
        maximum_prompt_tokens = max(maximum_prompt_tokens, observed_prompt_tokens)
        batch_size = int(llm.get("batch_size", 1))
        collate = _collator(tokenizer, max_length, torch)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loader = DataLoader(
            train_records,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
        )
        validation_loader = DataLoader(
            validation_records,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        if not trainable:
            raise ArtifactGateError("Qwen policy exposes no trainable LoRA/head parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(llm.get("learning_rate", 2e-4)),
            weight_decay=float(llm.get("weight_decay", 0.0)),
        )
        accumulation = int(llm.get("gradient_accumulation_steps", 16))
        if accumulation <= 0:
            raise ArtifactGateError("gradient_accumulation_steps must be positive")
        max_epochs = int(llm.get("max_epochs", distillation.get("max_epochs", 100)))
        patience_limit = int(
            llm.get(
                "early_stopping_patience",
                distillation.get("early_stopping_patience", 15),
            )
        )
        best_trainable = {
            name: parameter.detach().cpu().clone()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        best_epoch, best_validation, stale = 0, float("inf"), 0
        history: list[dict[str, float]] = []
        print(
            f"objective={objective} seed={seed} train_rows={len(train_records)} "
            f"validation_rows={len(validation_records)} prompt_tokens_max="
            f"{observed_prompt_tokens} max_epochs={max_epochs}",
            flush=True,
        )
        for epoch in range(1, max_epochs + 1):
            epoch_started = time.perf_counter()
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            train_total, train_soft, train_hard, train_seen = 0.0, 0.0, 0.0, 0
            for batch_index, batch in enumerate(train_loader, start=1):
                loss, soft_value, hard_value = _loss(
                    policy,
                    batch,
                    input_device=input_device,
                    head_device=head_device,
                    objective=objective,
                    temperature=temperature,
                    hard_weight=hard_weight,
                )
                (loss / accumulation).backward()
                if batch_index % accumulation == 0 or batch_index == len(train_loader):
                    clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                count = len(batch["input_ids"])
                train_total += float(loss.detach()) * count
                train_soft += soft_value * count
                train_hard += hard_value * count
                train_seen += count
                if batch_index % 500 == 0 or batch_index == len(train_loader):
                    print(
                        f"objective={objective} seed={seed} epoch={epoch} "
                        f"batch={batch_index}/{len(train_loader)} "
                        f"train_loss={train_total / max(train_seen, 1):.6f}",
                        flush=True,
                    )
            policy.eval()
            validation_total, validation_seen = 0.0, 0
            agreements = 0
            with torch.inference_mode():
                for batch in validation_loader:
                    loss, _, _, probabilities = _loss(
                        policy,
                        batch,
                        input_device=input_device,
                        head_device=head_device,
                        objective=objective,
                        temperature=temperature,
                        hard_weight=hard_weight,
                        return_probabilities=True,
                    )
                    predicted = probabilities.argmax(dim=-1).cpu()
                    agreements += int((predicted == batch["target_action"]).sum().item())
                    count = len(batch["input_ids"])
                    validation_total += float(loss) * count
                    validation_seen += count
            validation_loss = validation_total / max(validation_seen, 1)
            row = {
                "epoch": float(epoch),
                "loss": train_total / max(train_seen, 1),
                "soft_kl": train_soft / max(train_seen, 1),
                "hard_nll": train_hard / max(train_seen, 1),
                "validation_loss": validation_loss,
                "validation_top1_agreement": agreements / max(validation_seen, 1),
            }
            history.append(row)
            print(
                f"objective={objective} seed={seed} epoch={epoch} "
                f"validation_loss={validation_loss:.6f} "
                f"validation_top1_agreement={row['validation_top1_agreement']:.6f} "
                f"seconds={time.perf_counter() - epoch_started:.1f}",
                flush=True,
            )
            if validation_loss < best_validation - 1e-8:
                best_validation, best_epoch, stale = validation_loss, epoch, 0
                best_trainable = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in policy.named_parameters()
                    if parameter.requires_grad
                }
            else:
                stale += 1
                if stale >= patience_limit:
                    break
        _restore_trainable_parameters(policy, best_trainable, torch)
        checkpoint_dir = checkpoint_root / f"seed_{seed}"
        if checkpoint_dir.exists():
            raise ArtifactGateError(
                f"Refusing to merge a new Qwen payload into an existing directory: "
                f"{checkpoint_dir}; choose a fresh output directory"
            )
        policy.save_policy(
            checkpoint_dir,
            tokenizer=tokenizer,
            metadata={
                "schema_version": 1,
                "kind": "qwen_distilled_policy_checkpoint",
                "teacher_seed": seed,
                "objective": objective,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "fit_split": "train",
                "checkpoint_selection_split": "validation",
                "test_used_for_training_or_selection": False,
                "contains_true_label": False,
                "contains_hidden_modalities": False,
                "contains_oracle_q": False,
                "source_trajectories": str(trajectory_path),
                "source_trajectories_sha256": sha256(trajectory_path),
                "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
                **canonical_provenance,
                "source_prepared_sha256": trajectory_metadata.get(
                    "source_prepared_sha256"
                ),
                "source_preprocessor_sha256": trajectory_metadata.get(
                    "source_preprocessor_sha256"
                ),
                "base_model_id": model_id,
                "base_model_revision": resolved_revision,
                "base_model_snapshot": str(base_model_snapshot),
                "revision_resolution": revision_resolution,
                "trust_remote_code": trust_remote_code,
                "lora_target_modules": resolved_target_modules,
                "torch_dtype": torch_dtype,
                "training_device_map": training_device_map,
                "quantization": quantization_recipe,
                "history": history,
                "max_observed_prompt_tokens": observed_prompt_tokens,
                "max_length": max_length,
                "prompt_max_chars": prompt_max_chars,
                "training_budgets_usd": training_budgets,
                "budget_conditioned": True,
                "belief_mode": str(trajectory_metadata.get("belief_mode", "ordinal")),
                **teacher_provenance,
            },
        )
        metadata_file = checkpoint_dir / "policy_metadata.json"
        payload_digest, payload_files = directory_payload_sha256(
            checkpoint_dir, exclude={"policy_metadata.json"}
        )
        if "action_head.pt" not in payload_files or not any(
            name.startswith("qwen_adapter_or_backbone/") for name in payload_files
        ):
            raise ArtifactGateError(
                f"Qwen checkpoint {checkpoint_dir} lacks its action head or adapter payload"
            )
        policy_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        policy_metadata.update(
            {
                "checkpoint_payload_sha256": payload_digest,
                "checkpoint_payload_file_sha256": payload_files,
            }
        )
        write_json_atomic(metadata_file, policy_metadata)
        checkpoint_records.append(
            {
                "teacher_seed": seed,
                "path": str(checkpoint_dir),
                "metadata_sha256": sha256(metadata_file),
                "payload_sha256": payload_digest,
                "payload_file_sha256": payload_files,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
            }
        )
        print(
            f"seed={seed}: epoch={best_epoch}, validation_loss={best_validation:.6f}"
        )
        del optimizer, trainable, best_trainable
        del train_loader, validation_loader, collate
        del policy, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = checkpoint_root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "qwen_distilled_policy_manifest",
        "created_at_utc": utc_now(),
        "deployable": True,
        "privileged": False,
        "contains_true_label": False,
        "contains_hidden_modalities": False,
        "contains_oracle_q": False,
        "fit_split": "train",
        "checkpoint_selection_split": "validation",
        "held_out_split": "test",
        "test_used_for_training_or_selection": False,
        "actions": list(ACTIONS),
        "base_model": model_id,
        "base_model_id": model_id,
        "base_model_revision": resolved_revision,
        "base_model_snapshot": str(base_model_snapshot),
        "revision_resolution": revision_resolution,
        "trust_remote_code": trust_remote_code,
        "load_in_4bit": bool(llm.get("load_in_4bit", True)),
        "lora_target_modules": resolved_target_modules,
        "torch_dtype": torch_dtype,
        "training_device_map": training_device_map,
        "quantization": quantization_recipe,
        "structured_action_head": True,
        "max_observed_prompt_tokens": maximum_prompt_tokens,
        "max_length": max_length,
        "prompt_max_chars": prompt_max_chars,
        "objective": objective,
        "source_trajectories": str(trajectory_path),
        "source_trajectories_sha256": sha256(trajectory_path),
        "source_subset_sha256": trajectory_metadata.get("source_subset_sha256"),
        "split_manifest_sha256": trajectory_metadata.get("split_manifest_sha256"),
        **canonical_provenance,
        "source_prepared_sha256": trajectory_metadata.get("source_prepared_sha256"),
        "source_preprocessor_sha256": trajectory_metadata.get(
            "source_preprocessor_sha256"
        ),
        "training_budgets_usd": training_budgets,
        "budget_conditioned": True,
        "belief_mode": str(trajectory_metadata.get("belief_mode", "ordinal")),
        **teacher_provenance,
        "checkpoints": checkpoint_records,
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        metadata_path(manifest_path),
        {
            **manifest,
            "kind": "qwen_distilled_policy_manifest_metadata",
            "artifact_sha256": sha256(manifest_path),
        },
    )
    audit = write_audit(
        config,
        f"train_llm_student_{objective}",
        status="completed",
        inputs={"planner_trajectories": str(trajectory_path)},
        outputs={"checkpoint_dir": str(checkpoint_root), "manifest": str(manifest_path)},
        gates={
            "structured_ten_action_head": True,
            "invalid_actions_masked": True,
            "fit_split_train_only": True,
            "checkpoint_selection_validation_only": True,
            "test_excluded": True,
            "prompt_leakage_check": True,
            "stored_prompts_rebuilt_and_exactly_verified": True,
            "prompt_truncation_bound_to_verified_sidecar": True,
            "minimum_three_seeds_or_explicit_paired_seed42_ablation": True,
            "budget_conditioned_training": True,
            "immutable_base_model_revision": True,
            "full_checkpoint_payload_hashed": True,
        },
        details={
            "teacher_seeds": teacher_seeds,
            "paired_seed42_ablation": paired_seed42,
            "objective": objective,
            "base_model_revision": resolved_revision,
            "trust_remote_code": trust_remote_code,
        },
    )
    print(f"Wrote {len(checkpoint_records)} Qwen student checkpoints to {checkpoint_root}")
    print(f"Audit: {audit}")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.cache_subset_states import ArtifactGateError, load_config_file

    try:
        base_config = load_config_file(args.config)
        llm_config = base_config.get("llm")
        if not isinstance(llm_config, Mapping):
            raise ArtifactGateError("llm mapping is required")
        configured = str(
            llm_config.get(
                "train_objective",
                base_config.get("distillation", {}).get("objective", "soft_kl"),
            )
        )
        requested = llm_config.get("train_objectives")
        if requested is None:
            objectives = list(dict.fromkeys((configured, "soft_kl", "hard_action")))
        elif isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
            objectives = list(dict.fromkeys(str(value) for value in requested))
        else:
            raise ArtifactGateError("llm.train_objectives must be a sequence")
        unsupported = sorted(set(objectives) - {"soft_kl", "hard_action", "soft_kl_with_hard"})
        if unsupported:
            raise ArtifactGateError(f"Unsupported llm.train_objectives: {unsupported}")
        configured_output = str(base_config.get("artifacts", {}).get("llm_checkpoint_dir", ""))
        if not configured_output:
            raise ArtifactGateError("artifacts.llm_checkpoint_dir is required")
        manifests: list[Path] = []
        for objective in objectives:
            variant_config = copy.deepcopy(base_config)
            variant_config["llm"]["train_objective"] = objective
            output = Path(configured_output)
            if objective == "hard_action":
                output = output.with_name("qwen_hard")
            elif objective == "soft_kl_with_hard":
                output = output.with_name("qwen_soft_hard")
            variant_config["artifacts"]["llm_checkpoint_dir"] = str(output)
            manifests.append(run(variant_config))
        print("LLM variant manifests:")
        for manifest in manifests:
            print(f"- {manifest}")
    except (ArtifactGateError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"train_llm_student gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
