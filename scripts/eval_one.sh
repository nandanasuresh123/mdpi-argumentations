#!/usr/bin/env bash
set -e

MODEL="${1:-rnn}"        # simple | rnn | lstm | bert
DEVICE="${2:-cpu}"       # cpu | cuda
TOKENIZER="${3:-distilbert-base-uncased}"

python /app/src/model_training.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --tokenizer_name "$TOKENIZER" \
  --eval_only 1