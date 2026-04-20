Needs ffmpeg too

llama-server command:
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --fit-ctx 128000 \
  -np 1 \
  -fa on \
  -b 2048 \
  -ub 2048 \
  -ctk q8_0 \
  -ctv q8_0 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --reasoning-budget -1 \
  --chat-template-kwargs "{\"preserve_thinking\": true}"

llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q6_K_XL --no-mmproj -c 8192 --reasoning-budget 0.5
