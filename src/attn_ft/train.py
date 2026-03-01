from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
from huggingface_hub import HfApi
import shutil
import threading
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm.auto import tqdm
import os

from attn_ft.attn_hooks import AttnHookManager, extract_t2i_attn
from attn_ft.config import load_config
from attn_ft.data import AttnSupervisionCollator
from attn_ft.losses import *
from attn_ft.models import (
    filter_trainable_parameters,
    load_model_and_processor
)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    wandb_run = "wandb" if cfg.train.wandb_enabled else None
    
    DESIRED_EBS = 16
    BATCH_SIZE_PER_GPU = cfg.train.batch_size 
    NUM_GPUS = int(os.environ.get("WORLD_SIZE", 1))
    cfg.train.grad_accum_steps = DESIRED_EBS // (BATCH_SIZE_PER_GPU * NUM_GPUS)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with=wandb_run,
    )
    torch.manual_seed(cfg.train.seed)


    if cfg.train.wandb_enabled:
        wandb_kwargs = {
            "project_name": cfg.train.wandb_project,
            "config": {
                "loss": cfg.train.loss,
                "loss_weight": cfg.train.loss_weight,
                "batch_size": cfg.train.batch_size,
                "max_steps": cfg.train.max_steps,
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "warmup_steps": cfg.train.warmup_steps,
                "grad_accum_steps": cfg.train.grad_accum_steps,
                "model": cfg.model.name,
            },
        }
        if cfg.train.wandb_entity:
            wandb_kwargs["entity"] = cfg.train.wandb_entity
        if cfg.train.wandb_run_name:
            wandb_kwargs["name"] = cfg.train.wandb_run_name
        accelerator.init_trackers(**wandb_kwargs)

    model, processor = load_model_and_processor(
        cfg.model.name,
        cfg.model.trust_remote_code,
        cfg.model.load_in_4bit,
        cfg.model.lora_r,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
        cfg.model.lora_target_modules,
    )

    attn_manager = AttnHookManager()
    attn_manager.attach(model)

    dataset = load_dataset(cfg.dataset.hf_dataset_id, split="train")

    collator = AttnSupervisionCollator(processor=processor)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
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
        num_cycles=3
    )
    scheduler = accelerator.prepare(scheduler)
    
    if cfg.train.loss == "ce":
        attn_align_loss = ce_loss
        accelerator.print("Using cross-entropy loss for attention alignment")
    elif cfg.train.loss == "vacuum":
        attn_align_loss = vacuum_loss
        accelerator.print("Using vacuum loss for attention alignment")
    elif cfg.train.loss == "suppress":
        attn_align_loss = soft_suppression_loss
        accelerator.print("Using soft suppression loss for attention alignment")
    else:
        raise ValueError(f"Unsupported loss type: {cfg.train.loss}")

    model.train()
    step = 0
    progress = tqdm(
        total=total_training_steps,
        disable=not accelerator.is_main_process,
        desc="train",
    )
    for epoch in range(cfg.train.num_epochs):
        lm_loss_total = 0.0
        attn_loss_total = 0.0
        
        for batch in dataloader:
            if step >= total_training_steps:
                break

            with accelerator.accumulate(model):
                batch.inputs.to(accelerator.device)
                labels = batch.labels.to(accelerator.device)
                outputs = model(**batch.inputs, labels=labels)
                all_maps = attn_manager.get_attentions()
                attn_map = all_maps[-2]
                phrase_attn = extract_t2i_attn(attn_map, batch, processor)  # list[Tensor(head, #of text tok, L)]

                align_loss = attn_align_loss(phrase_attn, batch.masks)
                align_loss_item = align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss

                lm_loss_total += outputs.loss.item()
                attn_loss_total += align_loss_item * cfg.train.loss_weight
                # accelerator.print(
                #     f"step={step} lm_loss={outputs.loss.item():.4f} "
                #     f"attn_align_loss={align_loss_item:.4f} "
                # )
                loss = align_loss * cfg.train.loss_weight + outputs.loss

                attn_manager.clear()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    step += 1
                    progress.update(1)

                    if accelerator.is_main_process and step > 0 and step % cfg.train.log_every == 0:
                        avg_lm_loss = lm_loss_total / cfg.train.log_every
                        avg_attn_loss = attn_loss_total / cfg.train.log_every
                        avg_total_loss = avg_lm_loss + avg_attn_loss
                        if wandb_run is not None:
                            accelerator.log({
                                "lm_loss": avg_lm_loss,
                                "attn_align_loss": avg_attn_loss,
                                "total_loss": avg_total_loss,
                                "lr": scheduler.get_last_lr()[0],
                                },
                                step=step,
                            )
                        lm_loss_total = 0.0
                        attn_loss_total = 0.0

                    if accelerator.is_main_process and step > 0 and (step % cfg.train.save_every == 0 or step == total_training_steps-1) :
                        out_dir = Path(cfg.train.output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        staging_dir = out_dir / f"staging-{step}"
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(staging_dir)
                        metadata = gen_metadata(staging_dir, step, cfg.train, message=f"Checkpoint at step {step}")
                        
                        # print(f"[MAIN] Queueing Step {step} for upload.")
                        upload_queue.put((staging_dir, metadata))


        if step >= total_training_steps:
            break
    progress.close()
    accelerator.end_training()


        

upload_queue = queue.Queue()
api = HfApi()
REPO_ID = "Jackie2235/Qwen3-VL-8B-Instruct_attn_ft"
    
def upload_worker():
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
                repo_id=REPO_ID,
                commit_message=f"loss: {metadata['loss']}, loss_weight: {metadata['loss_weight']:.4f} - Step {metadata['step']}",
                repo_type="model"
            )
            # print(f"[UPLOADER] Step {step} uploaded successfully.")
            
            shutil.rmtree(folder_path)
            
        except Exception as e:
            print(f"[UPLOADER] Failed to upload Step {metadata['step']}: {e}")
            # Optional: You could re-add it to the queue to retry
            # upload_queue.put(task) 
        
        upload_queue.task_done()
        
def gen_metadata(staging_dir: str, current_step: int, train_cfg, message: str = ""):
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
    upload_thread = None
    if is_main_process:
        upload_thread = threading.Thread(target=upload_worker, daemon=False)
        upload_thread.start()

    train(args.config)

    if is_main_process and upload_thread is not None:
        upload_queue.join()
        upload_queue.put(None)
        # makes sure all uploads are done before exiting
        upload_thread.join()


if __name__ == "__main__":
    main()
