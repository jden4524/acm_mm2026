import argparse
from peft import PeftModel, LoraConfig
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration     
import os
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA weights into base model.")
    parser.add_argument(
        "--peft-dir",
        default="checkpoint-500",
        help="Checkpoint folder name to load (e.g., checkpoint-500).",
    )

    parser.add_argument(
        "--output-dir",
        default="hf_model-8B",
        help="Directory to save merged model.",
    )
    return parser.parse_args()


def load_and_save(adapter_path, output_path):
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    with open(f"{adapter_path}/metadata.json", "r") as f:
        meta = json.load(f)
    base_model_id = config["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(base_model_id)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_id,device_map="cpu",low_cpu_mem_usage=True,dtype="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    base_model_name = base_model_id.split("/")[-1]
    checkpoint_name = f"{base_model_name}_{meta['loss']}_{meta['step']}"
    checkpoint_dir = os.path.join(output_path, checkpoint_name)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(checkpoint_dir, safe_serialization=True) # Saves as .safetensors
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_name, checkpoint_dir
    

data_mapping = {
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMVP": {
                "class": "ImageMCQDataset",
                "dataset": "MMVP"
            },
            "BLINK": {
                "class": "ImageMCQDataset",
                "dataset": "BLINK"
            },
            "VStarBench": {
                "class": "ImageMCQDataset",
                "dataset": "VStarBench"
            },
            "NaturalBenchDataset": {
                "class": "ImageMCQDataset",
                "dataset": "NaturalBenchDataset"
            },
            "VisOnlyQA-VLMEvalKit": {
                "class": "ImageMCQDataset",
                "dataset": "VisOnlyQA-VLMEvalKit"
            },
            "MME": {
            "class": "ImageYORNDataset",
            "dataset": "MME"
            },
            "HallusionBench": {
            "class": "ImageYORNDataset",
            "dataset": "HallusionBench"
            },
            "POPE": {
            "class": "ImageYORNDataset",
            "dataset": "POPE"
            },
            "AMBER": {
            "class": "ImageYORNDataset",
            "dataset": "AMBER"
            },
            "GQA_TestDev_Balanced": {
                "class": "ImageVQADataset",
                "dataset": "GQA_TestDev_Balanced"
            }
}


def write_eval_json(checkpoint_name, checkpoint_dir):
    eval_config = {
        "model": {
            checkpoint_name: {
                "class": "Qwen3VLChat",
                "model_path": checkpoint_dir,
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

    fast_subset = ["HallusionBench", "MMVP", "VStarBench",  "VisOnlyQA-VLMEvalKit", "MME",  "POPE"]

    eval_config["data"] = {k: v for k, v in data_mapping.items() if k in fast_subset}
    
    with open("eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)
    
    
def main() -> None:
    args = parse_args()
    checkpoint_name, checkpoint_dir = load_and_save(args.peft_dir, args.output_dir)
    write_eval_json(checkpoint_name, checkpoint_dir)
    

if __name__ == "__main__":
    main()