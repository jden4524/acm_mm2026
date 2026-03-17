from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from huggingface_hub import snapshot_download
from pathlib import Path
import queue
import threading
import json
import subprocess
from log_to_wandb import log_eval_results
from prepare_eval_model import prepare_eval_model

app = FastAPI()
eval_queue = queue.Queue()
WEBHOOK_SECRET = "new checkpoint" # Match what you put in HF settings
BASE_DIR = Path(__file__).resolve().parent

def eval_worker():
    while True:
        # Get the (repo_id, commit_sha, commit_message) from the queue
        repo_id, commit_sha = eval_queue.get()
        print(f"--- [WORKER] Downloading {repo_id} with commit SHA: {commit_sha} ---")
        
        # 1. Download ONLY this specific version to a local folder
        local_dir = BASE_DIR / "eval_models" / "cache" / commit_sha
        merged_dir = BASE_DIR / "eval_models" / "merged"
        snapshot_download(repo_id=repo_id, revision=commit_sha, local_dir=local_dir)
        
        # 2. Run your evaluation script
        print(f"--- [WORKER] Converting PEFT weights to HF model for {repo_id} with commit SHA: {commit_sha} ---")

        checkpoint_name, checkpoint_dir = prepare_eval_model(local_dir, merged_dir)
        subprocess.run(["python", BASE_DIR / "VLMEvalKit" / "run.py", "--config", checkpoint_dir / "eval_config.json", "--work-dir", BASE_DIR / "eval_results"])
        subprocess.run(["rm", "-rf", checkpoint_dir])
        log_eval_results(checkpoint_name=checkpoint_name)
        eval_queue.task_done()
        
def get_checkpoint_notes(adapter_path):
    # You can use subprocess to call git log and get the commit message
    with open(f"{adapter_path}/metadata.json", "r") as f:
        meta = json.load(f)
    step = meta['step']
    return step


@app.post("/webhook")
async def hf_webhook(request: Request, background_tasks: BackgroundTasks):
    # Security: Check the secret header HF sends
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

    payload = await request.json()
    
    # Extract the Repo and the specific Commit SHA
    repo_id = payload['repo']['name']
    commit_sha = payload['repo']['headSha'] # This is the unique ID for THIS checkpoint
    
    print(f"[LISTENER] New commit detected: {commit_sha}")
    eval_queue.put((repo_id, commit_sha))
    
    return {"status": "queued"}

if __name__ == "__main__":
    # Start the worker thread
    threading.Thread(target=eval_worker, daemon=True).start()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
