cd $DATA_DIRECTORY
git clone https://github.com/jden4524/attn_ft.git
cd attn_ft
bash scripts/setup_env.sh
# bash scripts/train.sh || echo "Training failed, proceeding to destroy..."
# if [ -f "$DATA_DIRECTORY/keep_alive" ]; then
#     echo "Lock file found. Staying online."
# else
#     vastai destroy instance $CONTAINER_ID
# fi