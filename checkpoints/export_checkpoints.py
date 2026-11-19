"""Export the completed 4090pc study checkpoints from a trusted private training tree."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import joblib
import torch
import yaml


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def expect_hash(path, expected):
    check(digest(path) == expected, f"Historical SHA-256 mismatch: {path.name}")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def save_state(state, path):
    # Clone individual tensors so a storage view cannot carry unused bytes.
    plain = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
    torch.save(plain, path)
    restored = torch.load(path, weights_only=True, map_location="cpu")
    check(
        plain.keys() == restored.keys()
        and all(torch.equal(v, restored[k]) for k, v in state.items()),
        "Tensor values changed in export",
    )


def ordinal(root, out, ck, pp, cal, key, epoch, expected_sha, cfg_hashes):
    from src.ordinal.model import OrdinalBeliefModel

    expect_hash(ck, expected_sha)
    data = torch.load(ck, map_location="cpu", weights_only=True)
    expect_hash(pp, data["preprocessor_sha256"])
    expect_hash(cal, data["calibration_sha256"])
    for name in ("data", "ordinal"):
        check(
            data[f"{name}_config_sha256"] == cfg_hashes[f"configs/{name}.yaml"],
            "Ordinal config mismatch",
        )
    model = OrdinalBeliefModel(data["input_dims"], **data["model_config"])
    model.load_state_dict(data["state_dict"], strict=True)
    folder = out / "weights" / key
    folder.mkdir(parents=True)
    save_state(data["state_dict"], folder / "weights.pt")
    # joblib is used only on the trusted original, on the training host.
    prep = joblib.load(pp)
    check(prep.input_dims == data["input_dims"], "Encoder dimensions do not match model")
    specs = {m: {c: asdict(s) for c, s in cols.items()} for m, cols in prep.specs.items()}
    write_json(folder / "preprocessing.json", specs)
    calibration = read_json(cal)
    selected = {
        k: calibration[k]
        for k in ("method", "temperature", "calibrated_kappa1", "calibrated_kappa2")
    }
    check(
        selected["method"] == "ordinal_temperature_thresholds"
        and selected["temperature"] > 0
        and selected["calibrated_kappa1"] < selected["calibrated_kappa2"],
        "Invalid calibration",
    )
    write_json(folder / "calibration.json", selected)
    return {
        "branch": key,
        "seed": data["seed"],
        "best_epoch": epoch,
        "configs": ["configs/data.yaml", "configs/ordinal.yaml"],
        "directory": str(folder.relative_to(out)),
        "source": str(ck.relative_to(root)),
        "source_sha256": expected_sha,
        "source_preprocessor_sha256": digest(pp),
        "source_calibration_sha256": digest(cal),
        "input_dims": data["input_dims"],
        "model_config": data["model_config"],
    }


def qwen(folder, source, entry, model_info, llm):
    from safetensors import safe_open

    expect_hash(source / "policy_metadata.json", entry["metadata_sha256"])
    meta = read_json(source / "policy_metadata.json")
    check(
        meta["teacher_seed"] == entry["teacher_seed"]
        and meta["best_epoch"] == entry["best_epoch"]
        and meta["actions"] == model_info["actions"]
        and meta["objective"] == "soft_kl",
        "Qwen metadata mismatch",
    )
    check(
        meta["checkpoint_payload_file_sha256"] == entry["payload_file_sha256"],
        "Qwen file inventory mismatch",
    )
    for name, expected in entry["payload_file_sha256"].items():
        expect_hash(source / name, expected)
    adapter = "qwen_adapter_or_backbone"
    (folder / adapter).mkdir()
    original = source / adapter / "adapter_model.safetensors"
    with safe_open(original, framework="pt", device="cpu") as f:
        check(
            all(re.fullmatch(r"[.\w]+\.lora_[AB]\.weight", k) for k in f.keys())
            and f.metadata() in (None, {}, {"format": "pt"}),
            "Unexpected adapter tensors or metadata",
        )
    shutil.copyfile(original, folder / adapter / original.name)
    config = read_json(source / adapter / "adapter_config.json")
    check(
        config["r"] == llm["lora_rank"]
        and config["lora_alpha"] == llm["lora_alpha"]
        and config["lora_dropout"] == llm["lora_dropout"],
        "Qwen LoRA config mismatch",
    )
    check(
        meta["base_model_id"] == llm["model"] and meta["base_model_revision"] == llm["revision"],
        "Base revision mismatch",
    )
    config.update(base_model_name_or_path=llm["model"], revision=llm["revision"])
    write_json(folder / adapter / "adapter_config.json", config)
    head = torch.load(source / "action_head.pt", weights_only=True, map_location="cpu")
    check(
        set(head) == {"classifier.weight", "classifier.bias"}
        and head["classifier.weight"].shape == (len(meta["actions"]), meta["hidden_size"]),
        "Unexpected action head",
    )
    save_state(head, folder / "action_head.pt")
    write_json(
        folder / "policy_metadata.json",
        {"actions": meta["actions"], "hidden_size": meta["hidden_size"]},
    )
    return {
        "base_model": llm["model"],
        "revision": llm["revision"],
        "max_length": meta["max_length"],
        "prompt_max_chars": meta["prompt_max_chars"],
        "source_sha256": entry["metadata_sha256"],
        "source_payload_sha256": entry["payload_sha256"],
    }


def build(root, out):
    from scripts.cache_subset_states import load_config_file, planner_contract
    from src.planner.ebm import EnergyNetwork
    from src.students.mlp_policy import MLPPolicy

    art = root / "artifacts"
    files = {}
    for name in ("data", "ordinal", "planner", "costs", "distillation"):
        path = Path(f"configs/{name}.yaml")
        (out / path.parent).mkdir(exist_ok=True)
        shutil.copyfile(root / path, out / path)
        files[str(path)] = digest(root / path)
    cfg = yaml.safe_load((out / "configs/distillation.yaml").read_text())
    planner_root = art / "planner/checkpoints"
    teacher_manifest = read_json(planner_root / "manifest.json")
    check(
        teacher_manifest["planner_contract"]
        == planner_contract(load_config_file(root / "configs/planner.yaml")),
        "Historical planner config mismatch",
    )
    ord_root = art / "ordinal"
    summary = read_json(ord_root / "training_summary.json")
    entries = [
        ordinal(
            root,
            out,
            ord_root / "ordinal_belief.pt",
            art / "data/preprocessor.joblib",
            ord_root / "temperature_calibration.json",
            "ordinal",
            summary["best_epoch"],
            summary["checkpoint_sha256"],
            files,
        )
    ]
    check(
        entries[0]["source_sha256"] == teacher_manifest["ordinal_checkpoint_sha256"]
        and entries[0]["source_calibration_sha256"] == teacher_manifest["final_calibration_sha256"],
        "Wrong ordinal model or calibration for teacher",
    )
    folds = joblib.load(ord_root / "crossfit_beliefs.joblib")["metadata"]["fold_manifest"]
    check(sorted(f["fold"] for f in folds) == list(range(5)), "Expected five folds")
    for fold in folds:
        i = fold["fold"]
        source = ord_root / f"crossfit/fold_{i}"
        ck = source / "model.pt"
        expect_hash(ck, fold["model_sha256"])
        data = torch.load(ck, map_location="cpu", weights_only=True)
        check(data["seed"] == fold["training_seed"] and data["fold"] == i, "Fold mismatch")
        entries.append(
            ordinal(
                root,
                out,
                ck,
                source / "preprocessor.joblib",
                source / "temperature_calibration.json",
                f"ordinal_oof/fold_{i}",
                data["best_epoch"],
                fold["model_sha256"],
                files,
            )
        )
    teacher_sha = digest(planner_root / "manifest.json")
    trajectory_sha = digest(art / "trajectories/planner_records.parquet")
    trajectories = read_json(art / "trajectories/planner_records.parquet.metadata.json")
    check(
        trajectories["teacher_manifest_sha256"] == teacher_sha
        and trajectories["artifact_sha256"] == trajectory_sha,
        "Wrong trajectories for the selected teacher",
    )
    for branch in ("ebm", "mlp", "reactive_mlp", "qwen"):
        path = planner_root if branch == "ebm" else art / "students" / branch
        info = read_json(path / "manifest.json")
        if branch != "ebm":
            check(
                info["teacher_manifest_sha256"] == teacher_sha
                and info["source_trajectories_sha256"] == trajectory_sha,
                "Student does not match selected teacher/trajectories",
            )
        seed_key = "seed" if branch == "ebm" else "teacher_seed"
        check(
            sorted(e[seed_key] for e in info["checkpoints"]) == [42, 123, 2026],
            "Incomplete seed inventory",
        )
        for ent in info["checkpoints"]:
            seed = ent[seed_key]
            folder = out / "weights" / branch / f"seed_{seed}"
            folder.mkdir(parents=True)
            record = {
                "branch": branch,
                "seed": seed,
                "best_epoch": ent["best_epoch"],
                "configs": (
                    ["configs/planner.yaml", "configs/costs.yaml"]
                    if branch == "ebm"
                    else ["configs/distillation.yaml"]
                ),
                "directory": str(folder.relative_to(out)),
            }
            if branch == "qwen":
                source = path / f"seed_{seed}"
                record.update(qwen(folder, source, ent, info, cfg["llm"]))
                shared_tokenizer = folder.parent / "tokenizer"
                shared_tokenizer.mkdir(exist_ok=True)
                for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                    dest = shared_tokenizer / name
                    if not dest.exists():
                        shutil.copyfile(source / "tokenizer" / name, dest)
                    expect_hash(dest, ent["payload_file_sha256"][f"tokenizer/{name}"])
                record["tokenizer"] = str(shared_tokenizer.relative_to(out))
                source = source / "policy_metadata.json"
            else:
                source = path / f"seed_{seed}.pt"
                expect_hash(source, ent["sha256"])
                payload = torch.load(source, map_location="cpu", weights_only=True)
                check(
                    payload[seed_key] == seed
                    and payload["best_epoch"] == ent["best_epoch"]
                    and payload["actions"] == info["actions"],
                    "Seed, epoch or action order mismatch",
                )
                params = {k: payload[k] for k in ("input_dim", "hidden_dims", "dropout")}
                if branch == "ebm":
                    model = EnergyNetwork(
                        params["input_dim"], params["hidden_dims"], params["dropout"]
                    )
                    record["energy_temperature"] = payload["energy_temperature"]
                else:
                    params["activation"] = payload["activation"]
                    model = MLPPolicy.from_config(params)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                save_state(payload["model_state_dict"], folder / "weights.pt")
                record.update(model_config=params, source_sha256=ent["sha256"])
            record["source"] = str(source.relative_to(root))
            if branch != "ebm":
                record.update(teacher_seed=seed, objective=info["objective"])
            entries.append(record)
    for p in sorted((out / "weights").rglob("*")):
        if p.is_file():
            files[str(p.relative_to(out))] = digest(p)
    write_json(
        out / "manifest.json",
        {
            "format_version": 1,
            "run": "4090pc August 2026",
            "actions": teacher_manifest["actions"],
            "ordinal_labels": ["CN", "MCI", "AD"],
            "teacher_manifest_sha256": teacher_sha,
            "canonical_input_schema_sha256": teacher_manifest["h_input_schema_sha256"],
            "canonical_encoder_sha256": teacher_manifest["h_implementation_sha256"],
            "training_trajectories_sha256": trajectory_sha,
            "checkpoints": entries,
            "file_sha256": files,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    sys.path.insert(0, str(project))
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    build(project, out)


if __name__ == "__main__":
    main()
