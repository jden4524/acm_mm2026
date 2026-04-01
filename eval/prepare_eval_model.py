import argparse
import sys
from pathlib import Path
import json

from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

DATA_MAPPING_PATH = Path(__file__).with_name("data_mapping.json")

MODEL_FAMILIES = {
    "qwen3_vl": {
        "eval_class": "Qwen3VLChat",
        "eval_kwargs": {
            "use_custom_prompt": True,
            "max_new_tokens": 1024,
            "use_vllm": True,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_p": 0.8,
            "top_k": 20,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        "datasets": ["MMVP", "VStarBench", "VStarBench", "HallusionBench", "MME", "POPE", "VisOnlyQA-VLMEvalKit"],
        "processor_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True},
    },
    "llama_vision": {
        "eval_class": "llama_vision",
        "eval_kwargs": {
            "max_new_tokens": 1024,
            "temperature": 0.0,
        },
        "datasets": ["MMVP", "VStarBench"],
        "processor_kwargs": {},
        "model_kwargs": {},
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA weights into base model.")
    parser.add_argument(
        "--peft-dir",
        type=Path,
        default="checkpoint-500",
        help="Checkpoint folder name to load (e.g., checkpoint-500).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="hf_model",
        help="Directory to save merged model.",
    )
    return parser.parse_args()


def infer_model_family(base_model_id: str) -> str:
    model_name = base_model_id.lower()
    if "qwen3-vl" in model_name:
        return "qwen3_vl"
    if "llama-3.2" in model_name and "vision" in model_name:
        return "llama_vision"
    raise ValueError(f"Unsupported base model for eval preparation: {base_model_id}")


def load_and_save(adapter_path: Path, output_path: Path) -> tuple[str, Path, str]:
    with open(adapter_path /"adapter_config.json", "r") as f:
        config = json.load(f)
    with open(adapter_path /"metadata.json", "r") as f:
        meta = json.load(f)

    base_model_id = config["base_model_name_or_path"]
    model_family = infer_model_family(base_model_id)
    family_config = MODEL_FAMILIES[model_family]
    processor = AutoProcessor.from_pretrained(base_model_id, **family_config["processor_kwargs"])
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        **family_config["model_kwargs"],
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    checkpoint_name = f"{meta['message']}_{meta['step']}"
    checkpoint_dir = output_path / checkpoint_name
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(checkpoint_dir, safe_serialization=True) # Saves as .safetensors
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_name, checkpoint_dir, model_family
    

def load_data_mapping(mapping_path: Path = DATA_MAPPING_PATH) -> dict:
    with open(mapping_path, "r") as f:
        return json.load(f)


def write_eval_json(checkpoint_name: str, checkpoint_dir: Path, model_family: str) -> None:
    data_mapping = load_data_mapping()
    family_config = MODEL_FAMILIES[model_family]
    eval_config = {
        "model": {
            checkpoint_name: {
                "class": family_config["eval_class"],
                "model_path": str(checkpoint_dir),
                **family_config["eval_kwargs"],
            }
        }
    }

    subset_eval = family_config["datasets"]
    eval_config["data"] = {k: v for k, v in data_mapping.items() if k in subset_eval}
    config_path = checkpoint_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    
    
def prepare_eval_model(peft_dir: Path, output_dir: Path) -> tuple[str, Path]:
    checkpoint_name, checkpoint_dir, model_family = load_and_save(peft_dir, output_dir)
    write_eval_json(checkpoint_name, checkpoint_dir, model_family)
    return checkpoint_name, checkpoint_dir


if __name__ == "__main__":
    args = parse_args()
    prepare_eval_model(args.peft_dir, args.output_dir)