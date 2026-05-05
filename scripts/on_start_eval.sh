cd $DATA_DIRECTORY
git clone https://github.com/jden4524/attn_ft.git
cd attn_ft
cd eval
git clone https://github.com/hengzhan/VLMEvalKit.git
uv venv eval
source eval/bin/activate
uv pip install -e ./VLMEvalKit
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

ngrok config add-authtoken $NGROK_KEY
# Start ngrok in tmux
tmux new-session -d -s ngrok "ngrok http 8000"
# Start the evaluation server in same tmux session but different window
tmux new-window -t ngrok:1 -n "eval_server" "python watch.py"
