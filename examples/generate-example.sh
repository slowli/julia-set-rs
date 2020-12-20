#!/usr/bin/env bash

FEATURES="--features=dyn_cpu_backend"
BACKEND=cpu

cargo run --release --example cli "$FEATURES" -- \
  --backend "$BACKEND" \
  --iter 100 --inf 7 \
  --size 960x480 --height 12.25 \
  --palette snow \
  --output tiles.png \
  "log(cosh(1.02i * z)) - 5.61 + 0.2i"
