import argparse
from peft import PeftModel, LoraConfig
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration     
import os
from pathlib import Path
import json

DATA_MAPPING_PATH = Path(__file__).with_name("data_mapping.json")

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
        default="hf_model-8B",
        help="Directory to save merged model.",
    )
    return parser.parse_args()


def load_and_save(adapter_path, output_path):
    with open(adapter_path /"adapter_config.json", "r") as f:
        config = json.load(f)
    with open(adapter_path /"metadata.json", "r") as f:
        meta = json.load(f)
    base_model_id = config["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(base_model_id)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_id,device_map="cpu",low_cpu_mem_usage=True,dtype="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    checkpoint_name = f"{meta['message']}_{meta['step']}"
    checkpoint_dir = output_path / checkpoint_name
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(checkpoint_dir, safe_serialization=True) # Saves as .safetensors
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_name, checkpoint_dir
    

def load_data_mapping(mapping_path: Path = DATA_MAPPING_PATH) -> dict:
    with open(mapping_path, "r") as f:
        return json.load(f)


def write_eval_json(checkpoint_name, checkpoint_dir):
    data_mapping = load_data_mapping()
    eval_config = {
        "model": {
            checkpoint_name: {
                "class": "Qwen3VLChat",
                "model_path": str(checkpoint_dir),
                "use_custom_prompt": True,
                "max_new_tokens": 1024,
                "use_vllm": True,
                "temperature": 0.7,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "top_p": 0.8,
                "top_k": 20,
                "min_pixels":256 * 28 * 28,
                "max_pixels":1280 * 28 * 28,
            }
        }
    }

    fast_subset = ["HallusionBench", "MMVP", "VStarBench",  "VisOnlyQA-VLMEvalKit", "MME", "POPE"]
    lavender_subset = ["AI2D_TEST", "CCBench", "MMBench_DEV_EN", "MMBench_DEV_EN_V11", "MMStar", "SEEDBench_IMG", "ScienceQA_VAL"] # "MMMU_DEV_VAL", 
    
    subset_eval = fast_subset
    eval_config["data"] = {k: v for k, v in data_mapping.items() if k in subset_eval}
    config_path = checkpoint_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    
    
def prepare_eval_model(peft_dir: Path, output_dir: Path) -> None:
    checkpoint_name, checkpoint_dir = load_and_save(peft_dir, output_dir)
    write_eval_json(checkpoint_name, checkpoint_dir)
    return checkpoint_name, checkpoint_dir


if __name__ == "__main__":
    args = parse_args()
    prepare_eval_model(args.peft_dir, args.output_dir)