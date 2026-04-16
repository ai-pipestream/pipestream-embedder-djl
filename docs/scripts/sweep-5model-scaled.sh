#!/usr/bin/env bash
#
# Per-model worker scaling sweep for the 5 benchmark models on DJL Serving.
# Unlike run-djl-benchmark.sh which just runs every READY model with defaults,
# this one:
#   1. scales the target model up to a model-specific worker count
#   2. runs the harness at --concurrency 128 (Python path needs high c to
#      amortise DJL's ~600 ms per-request dispatch overhead)
#   3. scales it back to 1 worker before moving to the next model
#
# Why per-model worker counts: the 4080 Super has 16 GB and five models loaded
# simultaneously don't all get to scale to 16 workers. Budget roughly by
# parameter count — minilm (22 M) takes 16, bge-m3 (568 M) takes 2.
#
# Usage:  DJL_URL=http://djl-serving-host:8080 ./sweep-5model-scaled.sh
set -eo pipefail

DJL_URL="${DJL_URL:-http://localhost:8080}"

# WARNING: if you're hitting this from another host, use the direct LAN IP —
# routing through a Tailscale or WireGuard overlay adds 100-200x latency for
# medium-sized HTTP response bodies on this workload.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GRADLEW="$REPO_ROOT/gradlew"
OUTPUT="${OUTPUT:-build/bench/sweep-5model-djl.csv}"
CORPUS_LIMIT="${CORPUS_LIMIT:-100}"
BATCH="${BATCH:-32}"
NOTES_PREFIX="${NOTES_PREFIX:-djl-sweep}"

# name|dim|label_prefix|workers|concurrency
SPECS=(
  "all-MiniLM-L6-v2|384|minilm|16|128"
  "e5-small-v2|384|e5_small|8|128"
  "all-mpnet-base-v2|768|mpnet|4|128"
  "e5-large-v2|1024|e5_large|2|128"
  "bge-m3|1024|bge_m3|2|128"
)

scale_model() {
  local model="$1"; local n="$2"
  curl -sS -X PUT "$DJL_URL/models/$model?min_worker=$n&max_worker=$n" >/dev/null
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r model dim prefix workers conc <<< "$spec"
  label="${prefix}-djl-batch${BATCH}-w${workers}c${conc}"
  echo "================================================================"
  echo "=== $model  dim=$dim  workers=$workers  conc=$conc"
  echo "================================================================"

  echo "--- scaling $model to $workers workers"
  scale_model "$model" "$workers"
  sleep 8

  echo "--- running benchmark"
  ( cd "$REPO_ROOT" && "$GRADLEW" :embedder-test-harness:runDjl --no-daemon --quiet --console=plain \
      --args="--url $DJL_URL --model $model --dim $dim \
              --batch $BATCH --warmup 3 \
              --corpus-limit $CORPUS_LIMIT \
              --concurrency $conc \
              --label $label \
              --notes ${NOTES_PREFIX}-w${workers}c${conc} \
              --output $OUTPUT" ) || echo "!!! $model benchmark FAILED"

  echo "--- scaling $model back to 1 worker"
  scale_model "$model" 1
  sleep 5
done

echo "================================================================"
echo "=== sweep complete — results in $REPO_ROOT/embedder-test-harness/$OUTPUT"
echo "================================================================"
