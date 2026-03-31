from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
import queue
from typing import Iterator
from huggingface_hub import HfApi
import shutil
import threading
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm.auto import tqdm
import os

from attn_ft.attn_hooks import (
    AttnHookManager,
    extract_t2i_attn_valid,
    select_grounding_heads,
    update_head_stats,
)
from attn_ft.config import load_config
from attn_ft.data import AttnSupervisionCollator
from attn_ft.losses import soft_suppression_loss
from attn_ft.models import (
    filter_trainable_parameters,
    load_model_and_processor
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
def train(config_path: str) -> None:
    cfg = load_config(config_path)

    layer_schedule_cache: dict[int, tuple[list[int], list[float]]] = {}

    def build_layer_schedule(num_layers: int) -> tuple[list[int], list[float]]:
        if num_layers in layer_schedule_cache:
            return layer_schedule_cache[num_layers]

        if num_layers <= 0:
            layer_schedule_cache[num_layers] = ([], [])
            return [], []

        if cfg.model.guide_layers == "midlate":
            keep_count = max(1, math.ceil(num_layers * 0.4))
            start_idx = num_layers - keep_count
            layer_ids = list(range(start_idx, num_layers))
        elif cfg.model.guide_layers == "earlymid":
            keep_count = max(1, math.ceil(num_layers * 0.5))
            end_idx = keep_count
            layer_ids = list(range(end_idx))
        else:
            layer_ids = list(range(num_layers))

        raw_weights = []
        for layer_idx in layer_ids:
            distance_from_last = (num_layers - 1) - layer_idx
            raw_weights.append(math.exp(-cfg.model.attention_layer_decay * distance_from_last))

        weight_sum = sum(raw_weights)
        if weight_sum == 0:
            layer_weights = [1.0 / len(layer_ids)] * len(layer_ids)
            layer_schedule_cache[num_layers] = (layer_ids, layer_weights)
            return layer_ids, layer_weights

        layer_weights = [w / weight_sum for w in raw_weights]
        layer_schedule_cache[num_layers] = (layer_ids, layer_weights)
        return layer_ids, layer_weights

    model_cfg = AutoConfig.from_pretrained(cfg.model.name, trust_remote_code=True)
    if hasattr(model_cfg, "num_hidden_layers") and model_cfg.num_hidden_layers is not None:
        num_language_layers = int(model_cfg.num_hidden_layers)
    elif hasattr(model_cfg, "text_config") and getattr(model_cfg.text_config, "num_hidden_layers", None) is not None:
        num_language_layers = int(model_cfg.text_config.num_hidden_layers)
    else:
        raise ValueError(
            f"Could not infer number of language layers from config for model '{cfg.model.name}'"
        )

    layer_ids, _ = build_layer_schedule(num_language_layers)

    wandb_run = "wandb" if cfg.train.wandb_enabled else None
    
    NUM_GPUS = int(os.environ.get("WORLD_SIZE", 1))
    denom = cfg.train.micro_batch_size * NUM_GPUS
    cfg.train.grad_accum_steps = cfg.train.effective_batch_size // denom
    if cfg.train.grad_accum_steps < 1:
        raise ValueError(
            "effective_batch_size must be >= micro_batch_size * WORLD_SIZE "
            f"(got effective_batch_size={cfg.train.effective_batch_size}, "
            f"micro_batch_size={cfg.train.micro_batch_size}, WORLD_SIZE={NUM_GPUS})"
        )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with=wandb_run,
    )
    torch.manual_seed(cfg.train.seed)


    if cfg.train.wandb_enabled:
        wandb_kwargs = {
            "project_name": cfg.train.wandb_project,
            "init_kwargs": {
                "wandb": {"name": cfg.train.run_name, "job_type": "train"},
            },
            "config": {
                "loss": cfg.train.loss,
                "loss_weight": cfg.train.loss_weight,
                "micro_batch_size": cfg.train.micro_batch_size,
                "effective_batch_size": cfg.train.effective_batch_size,
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "warmup_steps": cfg.train.warmup_steps,
                "grad_accum_steps": cfg.train.grad_accum_steps,
                "model": cfg.model.name,
            },
        }
        accelerator.init_trackers(**wandb_kwargs)
    
    model, processor = load_model_and_processor(
        cfg.model.name,
        cfg.model.lora_r,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
        cfg.model.lora_target_modules,
        lora_layer_ids=layer_ids,
    )

    attn_manager = AttnHookManager()
    attn_manager.attach(model)

    loaded_datasets = []
    for dataset_id in cfg.dataset.hf_dataset_id:
        loaded = load_dataset(dataset_id, split=cfg.dataset.split)
        if "answer" not in loaded.column_names:
            loaded = loaded.rename_column("caption", "answer")
        if "question" not in loaded.column_names:
            loaded = loaded.add_column("question", ["Describe the image."] * len(loaded))
        loaded_datasets.append(loaded)
    if len(loaded_datasets) == 1:
        dataset = loaded_datasets[0]
    else:
        # sampling based on dataset sizes
        probabilities = [len(dset) for dset in loaded_datasets]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        dataset = interleave_datasets(loaded_datasets, probabilities=probabilities, stopping_strategy="first_exhausted")

    collator = AttnSupervisionCollator(processor=processor)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    trainable_dict = filter_trainable_parameters(model)
    trainable = list(trainable_dict.values())
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trainable)
        accelerator.print(f"total parameters: {total_params}")
        accelerator.print(f"trainable parameters: {trainable_params}")
    optimizer = AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    steps_per_epoch = len(dataloader) // cfg.train.grad_accum_steps
    total_training_steps = steps_per_epoch * cfg.train.num_epochs
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=total_training_steps,
        num_cycles=2
    )
    scheduler = accelerator.prepare(scheduler)
    
    if cfg.train.loss != "suppress":
        raise ValueError(f"Unsupported loss type: {cfg.train.loss}. This trainer only supports 'suppress'.")
    attn_align_loss = soft_suppression_loss
    accelerator.print("Using soft suppression loss for attention alignment")

    target_attn_layer = cfg.model.attention_layers
    if cfg.model.guide_layers == "all":
        accelerator.print(f"Using all attention layers with exponential decay={cfg.model.attention_layer_decay}")
    elif cfg.model.guide_layers == "midlate":
        accelerator.print(f"Using mid-to-late attention layers with exponential decay={cfg.model.attention_layer_decay}")
    # else:
    #     accelerator.print("Using attention layer index", target_attn_layer, "for supervision")

    if cfg.model.grounding_head_calibration:
        if accelerator.num_processes > 1:
            accelerator.print("Skipping grounding head calibration: prototype path supports single-process only")
        else:
            accelerator.print("Running grounding head calibration forward pass...")
            model.eval()
            head_stats = {}
            num_model_layers = len(accelerator.unwrap_model(model).model.model.language_model.layers)
            scheduled_layer_ids, _ = build_layer_schedule(num_model_layers)
            scheduled_layer_id_set = set(scheduled_layer_ids)
            calibration_iter = iter(dataloader)
            for _ in range(cfg.model.calibration_batches):
                batch = next(calibration_iter)

                batch.inputs.to(accelerator.device)
                try:
                    with torch.no_grad():
                        _ = model(**batch.inputs)

                    all_maps = attn_manager.get_attentions()
                    for layer_idx, attn_logits in enumerate(all_maps):
                        if layer_idx not in scheduled_layer_id_set:
                            continue
                        t2i_attn, valid_masks = extract_t2i_attn_valid(attn_logits, batch)
                        update_head_stats(head_stats, layer_idx, t2i_attn, valid_masks)
                        
                finally:
                    attn_manager.clear()

            grounding_heads, debug = select_grounding_heads(
                head_stats,
                allowed_layer_ids=scheduled_layer_ids,
            )
            accelerator.print(
                f"Grounding head calibration done: selected={debug.get('selected', 0)} / {debug.get('num_candidates', 0)}"
            )
            if debug.get("num_candidates", 0) > 0:
                accelerator.print(
                    f"Thresholds: mass>={debug['mass_thresh']:.6f} alignment<={debug['alignment_thresh']:.6f}"
                )
            accelerator.print("Selected grounding heads by layer:", grounding_heads)
            attn_manager.attach(model, selected_heads_map=grounding_heads)
            model.train()

    def build_accum_window(data_iterator: Iterator):
        micro_batches = []
        for _ in range(cfg.train.grad_accum_steps):
            try:
                micro_batches.append(next(data_iterator))
            except StopIteration:
                break

        local_tokens = sum((mb.labels[:, 1:] != -100).sum().item() for mb in micro_batches)
        local_attn_units = sum(len(mb.valid_supervision_indices) for mb in micro_batches)

        return micro_batches, local_tokens, local_attn_units

    model.train()
    step = 0
    progress = tqdm(
        total=total_training_steps,
        disable=not accelerator.is_main_process,
        desc="train",
    )
    log_steps = 0
    lm_metric_total = 0.0
    attn_metric_total = 0.0
    total_metric_total = 0.0
    for epoch in range(cfg.train.num_epochs):
        data_iter = iter(dataloader)
        while True:
            if step >= total_training_steps:
                break

            micro_batches, local_tokens, local_attn_units = build_accum_window(data_iter)

            if len(micro_batches) == 0:
                break

            local_tokens_t = torch.tensor([local_tokens], device=accelerator.device, dtype=torch.long)
            global_tokens = accelerator.gather(local_tokens_t).sum().clamp_min(1)
            local_attn_units_t = torch.tensor([local_attn_units], device=accelerator.device, dtype=torch.long)
            global_attn_units = accelerator.gather(local_attn_units_t).sum().clamp_min(1)

            window_lm_sum_local = torch.tensor(0.0, device=accelerator.device)
            window_attn_sum_local = torch.tensor(0.0, device=accelerator.device)
            # Process each micro-batch with accumulate()
            for batch in micro_batches:
                with accelerator.accumulate(model):
                    batch.inputs.to(accelerator.device)
                    labels = batch.labels.to(accelerator.device)

                    outputs = model(**batch.inputs)  # no labels; compute token loss manually
                    logits = outputs.logits

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    # sum over valid tokens
                    lm_loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )

                    lm_loss = (
                        lm_loss_sum
                        * accelerator.gradient_accumulation_steps
                        * accelerator.num_processes
                        / global_tokens
                    )

                    all_maps = attn_manager.get_attentions()
                    layer_ids, layer_weights = build_layer_schedule(len(all_maps))
                    
                    align_loss_sum_all_l = torch.tensor(0.0, device=accelerator.device)
                    pred_chunks: list[list[torch.Tensor]] | None = None
                    merged_masks: list[torch.Tensor] | None = None
                    head_weight_chunks: list[torch.Tensor] = []

                    for attn_layer, weight in zip(layer_ids, layer_weights):
                        layer_attn = all_maps[attn_layer]
                        if layer_attn.shape[1] == 0:
                            continue

                        pred_list, mask_list = extract_t2i_attn_valid(layer_attn, batch)
                        if not pred_list:
                            continue

                        heads_in_layer = pred_list[0].shape[0]
                        if heads_in_layer == 0:
                            continue

                        if pred_chunks is None:
                            pred_chunks = [[] for _ in pred_list]
                            merged_masks = mask_list

                        for sample_idx, sample_pred in enumerate(pred_list):
                            pred_chunks[sample_idx].append(sample_pred)

                        per_head_weight = weight / heads_in_layer
                        head_weight_chunks.append(
                            torch.full((heads_in_layer,), per_head_weight, device=accelerator.device, dtype=torch.float32)
                        )

                    if pred_chunks and merged_masks and head_weight_chunks:
                        merged_preds = [torch.cat(chunks, dim=0) for chunks in pred_chunks]
                        per_head_loss = attn_align_loss(merged_preds, merged_masks, per_head=True, temp=2)
                        head_weights = torch.cat(head_weight_chunks).to(per_head_loss.dtype)
                        align_loss_sum_all_l = (per_head_loss * head_weights.unsqueeze(0)).sum()
                    
                    align_loss = (
                        align_loss_sum_all_l
                        * accelerator.gradient_accumulation_steps
                        * accelerator.num_processes
                        / global_attn_units
                    )

                    window_lm_sum_local += lm_loss_sum.detach()
                    window_attn_sum_local += align_loss_sum_all_l.detach()

                    loss = lm_loss + cfg.train.loss_weight * align_loss

                    attn_manager.clear()
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    # gather true window sums for metrics
                    window_lm_sum_global = accelerator.gather(window_lm_sum_local.unsqueeze(0)).sum()
                    window_attn_sum_global = accelerator.gather(window_attn_sum_local.unsqueeze(0)).sum()

                    lm_metric = (window_lm_sum_global / global_tokens).item()
                    attn_metric = (window_attn_sum_global / global_attn_units).item()
                    total_metric = lm_metric + cfg.train.loss_weight * attn_metric

                    lm_metric_total += lm_metric
                    attn_metric_total += attn_metric
                    total_metric_total += total_metric
                    log_steps += 1

                    step += 1
                    progress.update(1)

                    if accelerator.is_main_process and step % cfg.train.log_every == 0 and wandb_run is not None:
                        avg_lm = lm_metric_total / log_steps
                        avg_attn = attn_metric_total / log_steps
                        avg_total = total_metric_total / log_steps
                        accelerator.log(
                            {
                                "lm_loss": avg_lm,
                                "attn_align_loss": avg_attn,  # unweighted
                                "total_loss": avg_total,
                                "lr": scheduler.get_last_lr()[0],
                            },
                            step=step,
                        )
                        lm_metric_total = 0.0
                        attn_metric_total = 0.0
                        total_metric_total = 0.0
                        log_steps = 0

                    if accelerator.is_main_process and step > 0 and (step % cfg.train.save_every == 0 or step == total_training_steps) :
                        out_dir = Path(cfg.train.run_name)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        staging_dir = out_dir / f"staging-{step}"
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(staging_dir)
                        metadata = gen_metadata(staging_dir, step, cfg.train, message=cfg.train.run_name)
                        
                        # print(f"[MAIN] Queueing Step {step} for upload.")
                        upload_queue.put((staging_dir, metadata))


        if step >= total_training_steps:
            break
    progress.close()
    accelerator.end_training()


        

upload_queue = queue.Queue()
api = HfApi()
    
def upload_worker(repo):
    """Background worker that processes the upload queue one by one."""
    while True:
        # Get upload task (folder_path, step_number)
        task = upload_queue.get()
        if task is None:
            upload_queue.task_done()
            break  # Graceful shutdown signal
        
        folder_path, metadata = task
        # print(f"[UPLOADER] Starting upload for Step {step}...")
        
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo,
                commit_message=f"loss: {metadata['loss']}, run name: {metadata['message']}, Step {metadata['step']}",
                repo_type="model"
            )
            # print(f"[UPLOADER] Step {step} uploaded successfully.")
            
            shutil.rmtree(folder_path)
            
        except Exception as e:
            print(f"[UPLOADER] Failed to upload Step {metadata['step']}: {e}")
            # Optional: You could re-add it to the queue to retry
            # upload_queue.put(task) 
        
        upload_queue.task_done()


def gen_metadata(staging_dir: str | Path, current_step: int, train_cfg, message: str = ""):
    metadata = {
        "loss": train_cfg.loss,
        "loss_weight": train_cfg.loss_weight,
        "step": current_step,
        "message": message
    }

    # Save this into the same staging folder you're uploading
    with open(f"{staging_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
    return metadata

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    is_main_process = int(os.environ.get("RANK", "0")) == 0
    if "qwen" in args.config.lower():
        REPO_ID = "Jackie2235/Qwen3-VL-8B-Instruct_attn_ft"
    elif "minicpm" in args.config.lower():
        REPO_ID = "Jackie2235/MiniCPM-attn_ft"
    else:
        raise ValueError(f"Cannot determine repo for config {args.config}")
    upload_thread = None
    if is_main_process:
        upload_thread = threading.Thread(target=upload_worker, daemon=False, args=(REPO_ID,))
        upload_thread.start()

    train(args.config)

    if is_main_process and upload_thread is not None:
        upload_queue.join()
        upload_queue.put(None)
        # makes sure all uploads are done before exiting
        upload_thread.join()


if __name__ == "__main__":
    main()
