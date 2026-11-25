# Shared by the H100 jobs. Everything lives under $PROJ so it can be removed after the run.
. /local/scratch/$USER/server_usage/scratch-env.sh
. /local/scratch/$USER/server_usage/require-compute.sh || exit 1
export PROJ=/local/scratch/$USER/prestige-mad
export HF_HOME="$PROJ/hf" HF_HUB_CACHE="$PROJ/hf/hub"
export VLLM_CACHE_ROOT="$PROJ/cache/vllm" UV_CACHE_DIR="$PROJ/cache/uv" UV_PYTHON_INSTALL_DIR="$PROJ/cache/python"
export VLLM_NO_USAGE_STATS=1
cd "$PROJ/code"
